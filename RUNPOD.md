# RunPod launch spec

## Pod
- **GPU**: 8× H100 80GB SXM, **secure cloud** (never community — see memory).
- **Image**: `runpod/pytorch:2.3.0-py3.10-cuda12.1.1-devel-ubuntu22.04`
- **Container disk**: **160 GB** (2× default per H100 rule; holds venv to avoid MooseFS slowness).
- **Volume `/workspace`**: **320 GB** (2× default for H100; checkpoints + shards + data).
  - If the allocated volume is MooseFS, venv must stay on container disk (already done in `pod_setup.sh`).
- **Ports**: 22 (SSH), 8888 (optional jupyter), 6006 (optional tensorboard).

## Cost envelope
- 8× H100 secure ≈ **€24/hr**
- SL warmstart ~8h, puzzle FT ~4h, RL self-play ~36h → **~48h × €24 ≈ €1,150**
- Buffer for eval + restarts: budget **€1,400** total.

## Bring-up (pod-side)
```bash
# Inside the pod, first time only:
bash /workspace/alphazero-chess/scripts/pod_setup.sh
# Then run the whole pipeline:
bash /workspace/alphazero-chess/scripts/pod_pipeline.sh
```

`pod_setup.sh` assumes the repo is already cloned at `/workspace/alphazero-chess`.
If you want the script to clone it itself, set `REPO_DIR=/workspace/alphazero-chess` and it
will clone from `github.com/cybercharlietang/alphazero-chess` on first run.

## Monitoring
- `nvidia-smi -l 2` on an SSH session.
- Optional: `wandb login` before launch; configs have `wandb` in deps.
- Checkpoints appear under `/workspace/alphazero-chess/runs/<stage>/`.

## Resume contract
All three configs read `resume_from:`. If a pod dies, on the next pod:
```
resume_from: runs/sl_warmstart/ckpt_00NNNNN.pt  # last written
```
and total_steps is respected, so training picks up.
