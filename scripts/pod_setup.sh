#!/usr/bin/env bash
# Run once on a fresh RunPod H100 pod to set up the environment.
# Assumes a pytorch:2.3+cuda12.1 base image. Put repo at /workspace/alphazero-chess.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/alphazero-chess}"
VENV_DIR="${VENV_DIR:-/root/venv}"   # container disk, avoids MooseFS slowness

echo "[1/6] apt install stockfish + build tools"
apt-get update -qq
apt-get install -y -qq stockfish zstd wget git build-essential >/dev/null

echo "[2/6] clone repo (skip if already present)"
if [[ ! -d "$REPO_DIR" ]]; then
  git clone https://github.com/cybercharlietang/alphazero-chess "$REPO_DIR"
fi
cd "$REPO_DIR"

echo "[3/6] create venv on container disk"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel -q

echo "[4/6] install pytorch + deps"
pip install -q torch --index-url https://download.pytorch.org/whl/cu121
pip install -q python-chess numpy tqdm zstandard pyyaml wandb pytest

echo "[5/6] download data"
mkdir -p data/raw
bash scripts/download_data.sh data/raw

echo "[6/6] verify stockfish + torch"
stockfish -h | head -1 || true
python -c "import torch; print('cuda available:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"
python -c "from alphazero.model import AlphaZeroNet, ModelConfig; m = AlphaZeroNet(ModelConfig()); print(f'model ok: {m.num_parameters()/1e6:.2f}M params')"

echo "Setup complete. Next: process data shards, then launch training."
