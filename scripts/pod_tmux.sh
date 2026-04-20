#!/usr/bin/env bash
# Launch the full training pipeline in a persistent tmux session with split
# panes for monitoring. SSH disconnect won't kill anything.
#
#   ssh to pod -> bash scripts/pod_tmux.sh     (starts session, detaches you)
#   reattach:  tmux attach -t alphacharles
#   detach:    ctrl-b then d
#   switch panes: ctrl-b then arrow
#
# Panes (in one window):
#   0 (left-top):    training log (pipeline stdout)
#   1 (right-top):   nvidia-smi -l 2
#   2 (left-bot):    tail -f latest wandb run
#   3 (right-bot):   shell for ad-hoc commands

set -euo pipefail
REPO_DIR="${REPO_DIR:-/workspace/alphazero-chess}"
VENV_DIR="${VENV_DIR:-/root/venv}"
SESSION="${TMUX_SESSION:-alphacharles}"

# tmux is pre-installed on most PyTorch images; install if missing.
if ! command -v tmux >/dev/null; then
  apt-get update -qq && apt-get install -y -qq tmux >/dev/null
fi

# If session already exists, just reattach.
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already running. Attach with: tmux attach -t $SESSION"
  exit 0
fi

cd "$REPO_DIR"

# Create session in detached mode with the training pipeline in pane 0.
tmux new-session -d -s "$SESSION" -n main \
  "source $VENV_DIR/bin/activate && cd $REPO_DIR && \
   bash scripts/pod_pipeline.sh 2>&1 | tee -a runs/pipeline.log"

# Split right: nvidia-smi.
tmux split-window -h -t "$SESSION:main" "nvidia-smi -l 2"

# Split bottom-left: wandb log tail.
tmux select-pane -t "$SESSION:main.0"
tmux split-window -v -t "$SESSION:main.0" \
  "while true; do \
     f=\$(ls -t $REPO_DIR/wandb/latest-run/logs/debug.log 2>/dev/null | head -1); \
     if [[ -n \"\$f\" ]]; then tail -F \"\$f\"; break; fi; \
     sleep 2; \
   done"

# Split bottom-right: free shell.
tmux select-pane -t "$SESSION:main.1"
tmux split-window -v -t "$SESSION:main.1" \
  "source $VENV_DIR/bin/activate && cd $REPO_DIR && bash"

# Even layout.
tmux select-layout -t "$SESSION:main" tiled
tmux select-pane -t "$SESSION:main.0"

echo ""
echo "Started tmux session: $SESSION"
echo "  Attach:        tmux attach -t $SESSION"
echo "  Detach:        ctrl-b then d"
echo "  Switch panes:  ctrl-b then arrow key"
echo "  Kill session:  tmux kill-session -t $SESSION  (stops training!)"
