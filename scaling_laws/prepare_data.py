"""
Prepare a fixed byte-corpus and tokenize it with multiple BPE tokenizers.

Pipeline:
  1. Download a slice of OpenWebText.
  2. Save raw text to a single file (the canonical byte-corpus).
  3. Train one BPE tokenizer per target vocab_size.
  4. For each tokenizer, write train.bin and val.bin (uint16 / uint32 token streams)
     plus a meta.json with vocab_size and bytes_per_token.

The same canonical byte-corpus is used everywhere so that comparing models trained
with different V is a fair comparison: they all see the same underlying text.
"""
import os
import json
import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

from tokenizer_utils import train_bpe, measure_bytes_per_token


def build_byte_corpus(out_path: str, target_bytes: int = 1_000_000_000):
    """Download OpenWebText and concatenate raw text until target_bytes is reached."""
    if os.path.exists(out_path) and os.path.getsize(out_path) >= target_bytes:
        print(f"Byte corpus already exists at {out_path} "
              f"({os.path.getsize(out_path)/1e9:.2f} GB)")
        return

    print(f"Building byte corpus ({target_bytes/1e9:.2f} GB target)...")
    ds = load_dataset("openwebtext", split="train", streaming=True,
                      trust_remote_code=True)
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in ds:
            text = ex["text"] + "\n"
            f.write(text)
            written += len(text.encode("utf-8"))
            if written >= target_bytes:
                break
    print(f"Wrote {written/1e9:.2f} GB to {out_path}")


def tokenize_and_save(corpus_path: str, tokenizer_path: str, output_dir: str,
                      val_fraction: float = 0.001, dtype=np.uint32):
    """Tokenize the full byte-corpus with the given tokenizer, write train/val bins."""
    os.makedirs(output_dir, exist_ok=True)
    tok = Tokenizer.from_file(tokenizer_path)
    print(f"Tokenizing {corpus_path} with {tokenizer_path}...")

    # encode in chunks to avoid loading huge text into a single encode call
    chunk_size = 10 * 1024 * 1024  # 10 MB chunks
    all_ids = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            ids = tok.encode(chunk).ids
            all_ids.extend(ids)
            if len(all_ids) % (10_000_000) < chunk_size // 4:
                print(f"  ...{len(all_ids)/1e6:.1f}M tokens so far")

    arr = np.array(all_ids, dtype=dtype)
    n_val = int(len(arr) * val_fraction)
    train = arr[:-n_val]
    val = arr[-n_val:]

    train_path = os.path.join(output_dir, "train.bin")
    val_path = os.path.join(output_dir, "val.bin")
    train.tofile(train_path)
    val.tofile(val_path)

    bytes_per_token = measure_bytes_per_token(tokenizer_path,
                                              open(corpus_path).read(1_000_000))
    meta = {
        "vocab_size": tok.get_vocab_size(),
        "bytes_per_token": bytes_per_token,
        "n_train_tokens": int(len(train)),
        "n_val_tokens": int(len(val)),
        "tokenizer": os.path.basename(tokenizer_path),
        "dtype": str(dtype.__name__),
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  V={meta['vocab_size']}: train={len(train):,}, val={len(val):,}, "
          f"bytes/tok={bytes_per_token:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", type=str, default="data/raw_corpus.txt")
    parser.add_argument("--target_bytes", type=int, default=1_000_000_000)
    parser.add_argument("--vocab_sizes", type=int, nargs="+",
                        default=[1024, 8192, 32768])
    parser.add_argument("--data_root", type=str, default="data")
    args = parser.parse_args()

    Path(args.data_root).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.data_root, "tokenizers")).mkdir(exist_ok=True)

    # Step 1: byte corpus
    build_byte_corpus(args.corpus_path, args.target_bytes)

    # Step 2: train tokenizers
    for v in args.vocab_sizes:
        tok_path = os.path.join(args.data_root, "tokenizers", f"bpe_v{v}.json")
        if not os.path.exists(tok_path):
            train_bpe(args.corpus_path, v, tok_path)
        else:
            print(f"Tokenizer already exists: {tok_path}")

    # Step 3: tokenize the corpus with each
    for v in args.vocab_sizes:
        tok_path = os.path.join(args.data_root, "tokenizers", f"bpe_v{v}.json")
        out_dir = os.path.join(args.data_root, f"v{v}")
        if os.path.exists(os.path.join(out_dir, "train.bin")):
            print(f"Already tokenized: V={v}")
            continue
        # uint16 only fits up to 65535, so use uint32 for safety
        tokenize_and_save(args.corpus_path, tok_path, out_dir, dtype=np.uint32)
