#!/usr/bin/env bash
# Launch parallel RL: 1 trainer on GPU 0 + 7 workers on GPUs 1-7.
# Each runs in its own tmux window under session 'rl'.
set -euo pipefail
REPO_DIR="${REPO_DIR:-/workspace/AlphaCharles}"
VENV_DIR="${VENV_DIR:-/root/venv}"
CONFIG="${CONFIG:-configs/rl_parallel.yaml}"
cd "$REPO_DIR"

tmux kill-session -t rl 2>/dev/null || true

# Trainer window
tmux new-session -d -s rl -n trainer -c "$REPO_DIR" \
  "source $VENV_DIR/bin/activate && export PYTHONPATH=$REPO_DIR && \
   python -u scripts/rl_parallel.py --role trainer --gpu 0 --config $CONFIG \
   2>&1 | tee -a runs/rl_parallel/trainer.log"

# 7 worker windows on GPUs 1..7
for i in 1 2 3 4 5 6 7; do
  tmux new-window -t rl -n "w$i" -c "$REPO_DIR" \
    "source $VENV_DIR/bin/activate && export PYTHONPATH=$REPO_DIR && \
     python -u scripts/rl_parallel.py --role worker --gpu $i --worker-id $i --config $CONFIG \
     2>&1 | tee -a runs/rl_parallel/worker_$i.log"
done

echo "=== RL launched in tmux session 'rl' ==="
echo "  attach: tmux attach -t rl"
echo "  windows: trainer, w1, w2, ... w7"
tmux list-windows -t rl
