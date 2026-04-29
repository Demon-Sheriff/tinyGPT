#!/usr/bin/env bash
set -e

source .venv/bin/activate
PROJECT="tying-experiment"
# 1 GPU per run, batch_size=48 * block_size=1024 * grad_accum=10 = 491,520 tokens/iter
# same effective batch size as the 8-GPU config in train_gpt2.py

CUDA_VISIBLE_DEVICES=0 python train.py config/train_gpt2.py \
    --tying_state=tied \
    --batch_size=48 --gradient_accumulation_steps=10 \
    --wandb_project=$PROJECT --wandb_run_name=tied \
    --out_dir=out-tied --compile=True &

CUDA_VISIBLE_DEVICES=1 python train.py config/train_gpt2.py \
    --tying_state=untied \
    --batch_size=48 --gradient_accumulation_steps=10 \
    --wandb_project=$PROJECT --wandb_run_name=untied \
    --out_dir=out-untied --compile=True &

CUDA_VISIBLE_DEVICES=2 python train.py config/train_gpt2.py \
    --tying_state=split \
    --batch_size=48 --gradient_accumulation_steps=10 \
    --wandb_project=$PROJECT --wandb_run_name=split \
    --out_dir=out-split --compile=True &

echo "All 3 runs launched (GPUs 0,1,2). GPU 3 is free."
echo "Monitor at: https://wandb.ai — project: $PROJECT"
wait
echo "All runs complete."
