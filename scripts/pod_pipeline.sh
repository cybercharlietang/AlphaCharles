#!/usr/bin/env bash
# End-to-end training pipeline on an 8xH100 pod.
# Prereq: pod_setup.sh has been run.
set -euo pipefail

# Default REPO_DIR to the script's own parent dir (robust to repo folder name).
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${VENV_DIR:-/root/venv}"
cd "$REPO_DIR"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$REPO_DIR"
mkdir -p runs

N_GPUS="${N_GPUS:-8}"

# Sentinel files let us skip completed stages on restart.
mark_done() { touch "runs/.done_$1"; }
already_done() { [[ -f "runs/.done_$1" ]]; }
stage() {
  local name=$1; shift
  if already_done "$name"; then
    echo "=== [skip] $name already completed ==="
    return
  fi
  echo "=== [run] $name ==="
  "$@"
  mark_done "$name"
}

stage "data_sl" python -m alphazero.data_pgn \
  --pgn data/raw/*.pgn \
  --out data/sl_shards \
  --min-rating 2200 \
  --max-samples 15000000

stage "data_puzzles" python -m alphazero.data_puzzles \
  --csv data/raw/lichess_db_puzzle.csv \
  --out data/puzzle_shards \
  --max-samples 3000000

stage "sl_warmstart" torchrun --nproc_per_node="$N_GPUS" \
  scripts/train_sl.py --config configs/sl_warmstart.yaml

stage "sl_puzzles" torchrun --nproc_per_node="$N_GPUS" \
  scripts/train_sl.py --config configs/sl_puzzles.yaml

stage "rl_selfplay" torchrun --nproc_per_node="$N_GPUS" \
  scripts/train_rl.py --config configs/rl_selfplay.yaml

echo "=== Eval vs stockfish depth=5, 10, 15 ==="
for d in 5 10 15; do
  python scripts/eval_vs_stockfish.py \
    --ckpt runs/rl_selfplay/final.pt \
    --stockfish "$(which stockfish)" \
    --n-games 20 --our-sims 800 --opp-depth $d | tee "runs/eval_stockfish_d${d}.txt"
done

# Write a sentinel file so we know training + eval are complete and the pod is
# ready to accept interactive play sessions.
touch runs/TRAINING_COMPLETE

echo ""
echo "===================================================="
echo " Training complete. Pod is now IDLE and ready to play."
echo " SSH in and run:"
echo "   cd $REPO_DIR && source $VENV_DIR/bin/activate"
echo "   python scripts/play_human.py --ckpt runs/rl_selfplay/final.pt --games 10"
echo ""
echo " DO NOT terminate the pod until you've played your match."
echo "===================================================="

# Keep the script alive so systemd / the launching shell doesn't consider
# the job "done" and trigger any auto-stop. User decides when to terminate.
tail -f /dev/null
