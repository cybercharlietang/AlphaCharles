"""Training step: combined policy + value loss.

AlphaZero loss per sample:
    L = (v - z)^2  -  pi^T log p  +  c * ||theta||^2

where:
    z  = game outcome (or SL target value), in [-1, 1]
    v  = net's value prediction
    pi = MCTS visit distribution (or one-hot SL target), length 4672
    p  = net's policy softmax (masked to legal moves)
    c  = L2 weight decay (applied via optimizer weight_decay)

The policy cross-entropy term pulls the net toward the MCTS/SL target;
the value MSE term pulls the scalar prediction toward the actual outcome.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class TrainConfig:
    lr: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 2048
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    grad_clip: float = 5.0


def policy_value_loss(
    policy_logits: torch.Tensor,   # (B, 4672)
    value_pred: torch.Tensor,      # (B,)
    policy_target: torch.Tensor,   # (B, 4672) probabilities (sum to 1)
    value_target: torch.Tensor,    # (B,)
    legal_mask: torch.Tensor | None = None,  # (B, 4672) bool; optional
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Return (total_loss, metrics_dict).

    Cross-entropy is computed as -sum(pi * log p). Because policy_target may
    contain zeros (unvisited moves), 0 * log(0) = 0 by convention — use
    masked log_softmax so illegal positions get log p = -inf but their pi = 0.
    """
    if legal_mask is not None:
        neg_inf = torch.finfo(policy_logits.dtype).min
        policy_logits = policy_logits.masked_fill(~legal_mask, neg_inf)
    log_probs = F.log_softmax(policy_logits, dim=-1)           # (B, 4672)
    # 0 * -inf = nan; guard by masking the product.
    prod = policy_target * log_probs
    prod = torch.where(policy_target > 0, prod, torch.zeros_like(prod))
    policy_loss = -prod.sum(dim=-1).mean()

    value_loss = F.mse_loss(value_pred, value_target)

    total = policy_weight * policy_loss + value_weight * value_loss
    return total, {
        "loss/total": float(total.detach()),
        "loss/policy": float(policy_loss.detach()),
        "loss/value": float(value_loss.detach()),
    }


def train_step(
    net,
    optimizer,
    batch: dict,
    cfg: TrainConfig,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> dict:
    """One optimization step. `batch` is a dict with keys planes, policy, value,
    and optionally legal_mask. Returns metrics dict."""
    net.train()
    planes = batch["planes"].to(device, non_blocking=True)
    policy_target = batch["policy"].to(device, non_blocking=True)
    value_target = batch["value"].to(device, non_blocking=True)
    legal_mask = batch.get("legal_mask")
    if legal_mask is not None:
        legal_mask = legal_mask.to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)

    if scaler is not None:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            policy_logits, value_pred = net(planes)
            loss, metrics = policy_value_loss(
                policy_logits, value_pred, policy_target, value_target,
                legal_mask=legal_mask,
                policy_weight=cfg.policy_loss_weight,
                value_weight=cfg.value_loss_weight,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        policy_logits, value_pred = net(planes)
        loss, metrics = policy_value_loss(
            policy_logits, value_pred, policy_target, value_target,
            legal_mask=legal_mask,
            policy_weight=cfg.policy_loss_weight,
            value_weight=cfg.value_loss_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        optimizer.step()

    return metrics
