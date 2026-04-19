"""Verify the loss decreases when we overfit a tiny batch."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from alphazero.encoding import NUM_PLANES, POLICY_SIZE
from alphazero.model import AlphaZeroNet, ModelConfig
from alphazero.training import TrainConfig, policy_value_loss, train_step


def test_loss_decreases_on_tiny_batch():
    torch.manual_seed(0)
    net = AlphaZeroNet(ModelConfig(channels=16, num_blocks=2))
    device = torch.device("cpu")
    cfg = TrainConfig(lr=3e-3, batch_size=8)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    B = 8
    planes = torch.randn(B, NUM_PLANES, 8, 8)
    # One-hot policy on random index.
    policy = torch.zeros(B, POLICY_SIZE)
    idxs = torch.randint(0, POLICY_SIZE, (B,))
    policy[torch.arange(B), idxs] = 1.0
    value = torch.rand(B) * 2 - 1  # [-1, 1]
    batch = {"planes": planes, "policy": policy, "value": value}

    losses = []
    for _ in range(50):
        m = train_step(net, opt, batch, cfg, device)
        losses.append(m["loss/total"])
    assert losses[-1] < losses[0] * 0.5, f"loss failed to drop: {losses[0]:.3f} -> {losses[-1]:.3f}"


def test_policy_loss_ignores_zero_targets():
    """If target has zeros, those should not contribute NaN."""
    logits = torch.zeros(2, POLICY_SIZE, requires_grad=True)
    pi = torch.zeros(2, POLICY_SIZE)
    pi[:, 0] = 1.0
    v_pred = torch.zeros(2)
    v_tgt = torch.zeros(2)
    loss, _ = policy_value_loss(logits, v_pred, pi, v_tgt)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
