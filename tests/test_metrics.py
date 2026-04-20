"""Smoke test: MetricsTracker works without W&B and computes batch stats correctly."""

from __future__ import annotations

import os

import pytest
import torch

from alphazero.encoding import POLICY_SIZE
from alphazero.metrics import MetricsTracker


@pytest.fixture(autouse=True)
def disable_rank(monkeypatch):
    # Ensure we're "main rank" for these tests.
    monkeypatch.delenv("RANK", raising=False)


def test_tracker_no_wandb():
    t = MetricsTracker(run_name="test", wandb_enabled=False, stdout_every=9999)
    t.log({"loss/total": 1.0}, step=1)
    assert t.wandb is None
    t.finish()


def test_batch_stats_sane():
    torch.manual_seed(0)
    t = MetricsTracker(run_name="test", wandb_enabled=False, stdout_every=9999)
    B = 32
    logits = torch.randn(B, POLICY_SIZE)
    # One-hot targets.
    pt = torch.zeros(B, POLICY_SIZE)
    idx = torch.randint(0, POLICY_SIZE, (B,))
    pt[torch.arange(B), idx] = 1.0
    v = torch.rand(B) * 2 - 1
    z = torch.rand(B) * 2 - 1
    m = t.log_batch_stats(logits, v, pt, z, step=1)
    assert 0 < m["policy/entropy"] < 20
    assert 0 <= m["policy/top1_acc"] <= 1
    assert -1 <= m["value/corr"] <= 1
    t.finish()


def test_grad_norm_reported():
    t = MetricsTracker(run_name="test", wandb_enabled=False, stdout_every=9999)
    m = torch.nn.Linear(4, 4)
    x = torch.randn(8, 4)
    loss = m(x).sum()
    loss.backward()
    gn = t.log_grad_norm(m, step=1, clip_value=5.0)
    assert gn > 0
    t.finish()
