#!/usr/bin/env bash
set -e

# --- 1. Install uv ---
if ! command -v uv &> /dev/null; then
    echo "=== Installing uv ==="
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
fi

# --- 2. Clone repo ---
cd $HOME
if [ ! -d "tinyGPT" ]; then
    echo "=== Cloning repo ==="
    git clone git@github.com:Demon-Sheriff/tinyGPT.git
fi
cd tinyGPT

# --- 3. Create venv and install dependencies ---
echo "=== Setting up Python environment ==="
uv venv --python 3.11
source .venv/bin/activate
uv pip install torch numpy tiktoken datasets wandb

# --- 4. Prepare OpenWebText data ---
if [ ! -f "data/openwebtext/train.bin" ]; then
    echo "=== Preparing OpenWebText data (this takes a while) ==="
    python data/openwebtext/prepare.py
else
    echo "=== OpenWebText data already prepared ==="
fi

# --- 5. wandb login ---
echo "=== Logging into wandb ==="
wandb login

echo ""
echo "=== Setup complete ==="
echo "To launch experiments, run inside a tmux session:"
echo "  tmux new -s train"
echo "  source .venv/bin/activate"
echo "  bash launch_experiments.sh"
