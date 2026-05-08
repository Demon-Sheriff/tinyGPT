#!/usr/bin/env bash
# Phase 1 pilot: 9 runs across 8 GPUs.
# Run 8 in parallel on GPUs 0-7, then 1 final on GPU 0 after the first batch.
set -uo pipefail

cd "$(dirname "$0")"
source ../.venv/bin/activate

PROJECT="dv-scaling-pilot"

run_one() {
    local gpu=$1
    local V=$2
    local D=$3
    local n_layer=$4
    local n_head=$5
    local flops=$6

    local name="V${V}-D${D}-L${n_layer}"
    local out_dir="out/${name}"
    local data_dir="data/v${V}"

    echo "[GPU ${gpu}] Launching ${name}..."
    CUDA_VISIBLE_DEVICES=${gpu} python train.py \
        --data_dir="${data_dir}" \
        --out_dir="${out_dir}" \
        --n_layer=${n_layer} --n_head=${n_head} --n_embd=${D} \
        --batch_size=32 --gradient_accumulation_steps=1 \
        --compute_budget_flops=${flops} \
        --wandb_log --wandb_project="${PROJECT}" --wandb_run_name="${name}" \
        > "logs/${name}.log" 2>&1 &
}

mkdir -p logs out

# Pilot grid: 3x3 = 9 runs
declare -a CONFIGS=(
    "1024 256 6 8 1e17"
    "1024 512 6 8 1e17"
    "1024 1024 6 8 1e17"
    "8192 256 6 8 1e17"
    "8192 512 6 8 1e17"
    "8192 1024 6 8 1e17"
    "32768 256 6 8 1e17"
    "32768 512 6 8 1e17"
    "32768 1024 6 8 1e17"
)

# Launch first 8 in parallel on GPUs 0-7
for i in $(seq 0 7); do
    run_one $i ${CONFIGS[$i]}
    sleep 2
done

echo "Launched 8 runs. Waiting for them to finish before launching the 9th..."
wait

# Final run on GPU 0
run_one 0 ${CONFIGS[8]}
wait

echo "All 9 runs complete."
echo "Run the analysis with: python analysis.py"
