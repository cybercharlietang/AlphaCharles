#!/usr/bin/env bash
# End-to-end training pipeline on an 8xH100 pod.
# Prereq: pod_setup.sh has been run.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/alphazero-chess}"
VENV_DIR="${VENV_DIR:-/root/venv}"
cd "$REPO_DIR"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$REPO_DIR"

N_GPUS="${N_GPUS:-8}"

echo "=== [1/5] Build SL shards from Lichess Elite PGNs ==="
python -m alphazero.data_pgn \
  --pgn data/raw/*.pgn \
  --out data/sl_shards \
  --min-rating 2200 \
  --max-samples 30000000

echo "=== [2/5] Build puzzle shards ==="
python -m alphazero.data_puzzles \
  --csv data/raw/lichess_db_puzzle.csv \
  --out data/puzzle_shards \
  --max-samples 3000000

echo "=== [3/5] SL warm-start ==="
torchrun --nproc_per_node="$N_GPUS" scripts/train_sl.py --config configs/sl_warmstart.yaml

echo "=== [4/5] Puzzle fine-tune ==="
torchrun --nproc_per_node="$N_GPUS" scripts/train_sl.py --config configs/sl_puzzles.yaml

echo "=== [5/5] Self-play RL ==="
torchrun --nproc_per_node="$N_GPUS" scripts/train_rl.py --config configs/rl_selfplay.yaml

echo "=== Done. Running eval vs stockfish depth=10 ==="
python scripts/eval_vs_stockfish.py \
  --ckpt runs/rl_selfplay/final.pt \
  --stockfish "$(which stockfish)" \
  --n-games 20 --our-sims 800 --opp-depth 10
