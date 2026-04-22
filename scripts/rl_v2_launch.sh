#!/usr/bin/env bash
# Launch parallel RL v2: 1 trainer on GPU 0 + 2 workers per GPU on 1..7.
# That's 14 workers total.
set -euo pipefail
REPO_DIR="${REPO_DIR:-/workspace/AlphaCharles}"
VENV_DIR="${VENV_DIR:-/root/venv}"
CONFIG="${CONFIG:-configs/rl_v2_test.yaml}"
cd "$REPO_DIR"

tmux kill-session -t rl 2>/dev/null || true

# Trainer on GPU 0
tmux new-session -d -s rl -n trainer -c "$REPO_DIR" \
  "source $VENV_DIR/bin/activate && export PYTHONPATH=$REPO_DIR && \
   mkdir -p runs/rl_v2_test && \
   python -u scripts/rl_parallel.py --role trainer --gpu 0 --config $CONFIG \
   2>&1 | tee -a runs/rl_v2_test/trainer.log"

# 2 workers per GPU on GPUs 1..7 = 14 workers
wid=0
for gpu in 1 2 3 4 5 6 7; do
  for slot in a b; do
    wid=$((wid + 1))
    tmux new-window -t rl -n "w${gpu}${slot}" -c "$REPO_DIR" \
      "source $VENV_DIR/bin/activate && export PYTHONPATH=$REPO_DIR && \
       mkdir -p runs/rl_v2_test && \
       python -u scripts/rl_parallel.py --role worker --gpu $gpu --worker-id $wid --config $CONFIG \
       2>&1 | tee -a runs/rl_v2_test/worker_${wid}.log"
  done
done

echo "=== RL v2 launched: 1 trainer + 14 workers (2/GPU) ==="
tmux list-windows -t rl | head -20
