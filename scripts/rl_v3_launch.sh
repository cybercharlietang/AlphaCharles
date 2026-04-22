#!/usr/bin/env bash
# Launch parallel RL v3. Workers-per-GPU controlled by $WPG env var (default 4).
set -euo pipefail
REPO_DIR="${REPO_DIR:-/workspace/AlphaCharles}"
VENV_DIR="${VENV_DIR:-/root/venv}"
CONFIG="${CONFIG:-configs/rl_v3.yaml}"
WPG="${WPG:-4}"   # workers per GPU
cd "$REPO_DIR"

tmux kill-session -t rl 2>/dev/null || true

# Trainer on GPU 0
tmux new-session -d -s rl -n trainer -c "$REPO_DIR" \
  "source $VENV_DIR/bin/activate && export PYTHONPATH=$REPO_DIR && \
   mkdir -p runs/rl_v3 && \
   python -u scripts/rl_parallel.py --role trainer --gpu 0 --config $CONFIG \
   2>&1 | tee -a runs/rl_v3/trainer.log"

# WPG workers per GPU on GPUs 1..7
SLOT_LETTERS=(a b c d e f g h i j k l m n o p)
wid=0
for gpu in 1 2 3 4 5 6 7; do
  for ((s=0; s<WPG; s++)); do
    wid=$((wid + 1))
    slot="${SLOT_LETTERS[s]}"
    tmux new-window -t rl -n "w${gpu}${slot}" -c "$REPO_DIR" \
      "source $VENV_DIR/bin/activate && export PYTHONPATH=$REPO_DIR && \
       python -u scripts/rl_parallel.py --role worker --gpu $gpu --worker-id $wid --config $CONFIG \
       2>&1 | tee -a runs/rl_v3/worker_${wid}.log"
  done
done

echo "=== RL v3 launched: 1 trainer + $((WPG * 7)) workers (${WPG}/GPU) ==="
