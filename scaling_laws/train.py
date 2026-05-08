"""
Training script for D/V scaling law experiments.

Key differences from the parent train.py:
  - Loads tokenized data from data/v{V}/, not data/openwebtext
  - Reads vocab_size and bytes_per_token from data/v{V}/meta.json
  - Logs bits-per-byte (BPB) on top of nat-loss for tokenizer-agnostic comparison
  - Run name encoded with V, D, n_layer for clear wandb tracking
  - Stops after a fixed compute budget (FLOPs) rather than fixed iterations,
    so different (V, D) combos all get the same compute
"""
import os
import time
import math
import json
import pickle
import argparse
from contextlib import nullcontext

import numpy as np
import torch

from model import GPTConfig, GPT


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True,
                   help="Path to data/v{V}/ with train.bin, val.bin, meta.json")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_embd", type=int, default=512)
    p.add_argument("--block_size", type=int, default=1024)
    p.add_argument("--bias", action="store_true", default=False)
    p.add_argument("--weight_tying", action="store_true", default=True)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=6e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_iters", type=int, default=100)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)

    p.add_argument("--compute_budget_flops", type=float, default=1e17,
                   help="Train until this many FLOPs are spent (6N FLOPs/token).")
    p.add_argument("--flops_definition", type=str, default="total",
                   choices=["total", "non_embedding"],
                   help="Whether N includes embedding params for FLOPs accounting.")

    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--log_interval", type=int, default=10)

    p.add_argument("--wandb_log", action="store_true", default=False)
    p.add_argument("--wandb_project", type=str, default="dv-scaling-laws")
    p.add_argument("--wandb_run_name", type=str, default=None)

    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--compile", action="store_true", default=True)
    p.add_argument("--no_compile", dest="compile", action="store_false")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load metadata for this tokenizer/vocab
    with open(os.path.join(args.data_dir, "meta.json")) as f:
        meta = json.load(f)
    vocab_size = meta["vocab_size"]
    bytes_per_token = meta["bytes_per_token"]
    print(f"Loaded {args.data_dir}: V={vocab_size}, bytes/token={bytes_per_token:.3f}")

    # Setup
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = "cuda" if "cuda" in args.device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
               "float16": torch.float16}[args.dtype]
    ctx = (nullcontext() if device_type == "cpu"
           else torch.amp.autocast(device_type=device_type, dtype=ptdtype))

    # Data loader: uint32 tokens (we used uint32 in prepare_data)
    np_dtype = np.uint32

    def get_batch(split):
        path = os.path.join(args.data_dir, f"{split}.bin")
        data = np.memmap(path, dtype=np_dtype, mode="r")
        ix = torch.randint(len(data) - args.block_size, (args.batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i + args.block_size].astype(np.int64))
                         for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + args.block_size].astype(np.int64))
                         for i in ix])
        if device_type == "cuda":
            x = x.pin_memory().to(args.device, non_blocking=True)
            y = y.pin_memory().to(args.device, non_blocking=True)
        return x, y

    # Model
    config = GPTConfig(
        block_size=args.block_size,
        vocab_size=vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=args.bias,
        weight_tying=args.weight_tying,
    )
    model = GPT(config).to(args.device)
    n_total = model.count_params(include_embeddings=True)
    n_non_emb = model.count_params(include_embeddings=False)
    n_for_flops = n_total if args.flops_definition == "total" else n_non_emb

    # Compute budget -> total tokens
    # FLOPs = 6 * N * tokens (Kaplan formula, fwd+bwd)
    total_tokens = int(args.compute_budget_flops / (6 * n_for_flops))
    tokens_per_iter = args.batch_size * args.block_size * args.gradient_accumulation_steps
    max_iters = max(1, total_tokens // tokens_per_iter)
    print(f"FLOPs budget: {args.compute_budget_flops:.2e} -> {total_tokens:,} tokens, "
          f"{max_iters:,} iters @ {tokens_per_iter} tok/iter (N={n_for_flops/1e6:.1f}M)")

    optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate,
                                           (args.beta1, args.beta2), device_type)

    if args.compile:
        print("compiling...")
        unoptimized_model = model
        model = torch.compile(model)
    raw_model = unoptimized_model if args.compile else model

    # LR schedule
    warmup = args.warmup_iters
    min_lr = args.learning_rate * args.min_lr_ratio
    decay_iters = max_iters
    def get_lr(it):
        if it < warmup:
            return args.learning_rate * (it + 1) / (warmup + 1)
        if it > decay_iters:
            return min_lr
        decay_ratio = (it - warmup) / max(1, decay_iters - warmup)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (args.learning_rate - min_lr)

    # Wandb
    use_wandb = args.wandb_log
    if use_wandb:
        import wandb
        run_name = args.wandb_run_name or f"V{vocab_size}-D{args.n_embd}-L{args.n_layer}"
        wandb_config = vars(args).copy()
        wandb_config.update({
            "vocab_size": vocab_size,
            "bytes_per_token": bytes_per_token,
            "n_total_params": n_total,
            "n_non_emb_params": n_non_emb,
            "n_for_flops": n_for_flops,
            "compute_budget_flops": args.compute_budget_flops,
            "total_tokens": total_tokens,
            "max_iters": max_iters,
            "v_over_d": vocab_size / args.n_embd,
        })
        wandb.init(project=args.wandb_project, name=run_name, config=wandb_config)

    # Eval
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(args.eval_iters)
            for k in range(args.eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    def loss_to_bpb(loss_nats):
        # loss is mean nats per token. Convert to bits per byte:
        # bpb = (loss * tokens_in_text / bytes_in_text) / ln(2)
        #     = (loss / bytes_per_token) / ln(2)
        return loss_nats / (bytes_per_token * math.log(2))

    # Training loop
    X, Y = get_batch("train")
    iter_num = 0
    t0 = time.time()
    while True:
        lr = get_lr(iter_num)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        if iter_num % args.eval_interval == 0:
            losses = estimate_loss()
            bpb_train = loss_to_bpb(losses["train"])
            bpb_val = loss_to_bpb(losses["val"])
            print(f"step {iter_num}: train {losses['train']:.4f} ({bpb_train:.4f} BPB), "
                  f"val {losses['val']:.4f} ({bpb_val:.4f} BPB)")
            if use_wandb:
                wandb.log({
                    "iter": iter_num,
                    "train/loss_nats": losses["train"],
                    "val/loss_nats": losses["val"],
                    "train/bpb": bpb_train,
                    "val/bpb": bpb_val,
                    "lr": lr,
                    "tokens_seen": iter_num * tokens_per_iter,
                    "flops_spent": 6 * n_for_flops * iter_num * tokens_per_iter,
                })

        for _ in range(args.gradient_accumulation_steps):
            with ctx:
                _, loss = model(X, Y)
                loss = loss / args.gradient_accumulation_steps
            X, Y = get_batch("train")
            loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        iter_num += 1
        if iter_num >= max_iters:
            break

    # Final eval
    losses = estimate_loss()
    bpb_val = loss_to_bpb(losses["val"])
    print(f"FINAL step {iter_num}: val {losses['val']:.4f} ({bpb_val:.4f} BPB)")
    if use_wandb:
        wandb.log({
            "iter": iter_num,
            "train/loss_nats": losses["train"],
            "val/loss_nats": losses["val"],
            "train/bpb": loss_to_bpb(losses["train"]),
            "val/bpb": bpb_val,
            "final/val_bpb": bpb_val,
            "final/val_nats": losses["val"],
        })
        wandb.summary["final_val_bpb"] = bpb_val
        wandb.summary["final_val_nats"] = losses["val"]
        wandb.summary["v_over_d"] = vocab_size / args.n_embd
        wandb.summary["n_total_params"] = n_total
        wandb.summary["n_non_emb_params"] = n_non_emb
        wandb.finish()

    # Persist final results to a JSON for offline analysis
    result = {
        "V": vocab_size,
        "D": args.n_embd,
        "n_layer": args.n_layer,
        "n_total_params": n_total,
        "n_non_emb_params": n_non_emb,
        "n_for_flops": n_for_flops,
        "flops_budget": args.compute_budget_flops,
        "total_tokens": total_tokens,
        "bytes_per_token": bytes_per_token,
        "v_over_d": vocab_size / args.n_embd,
        "final_val_loss_nats": losses["val"],
        "final_val_bpb": bpb_val,
        "wall_time_s": time.time() - t0,
    }
    with open(os.path.join(args.out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved result to {args.out_dir}/result.json")


if __name__ == "__main__":
    main()
