"""Smoke tests for the ResNet: shapes, param count, forward+backward."""

from __future__ import annotations

import chess
import numpy as np
import pytest
import torch

from alphazero.encoding import POLICY_SIZE, encode_board, legal_move_mask
from alphazero.model import AlphaZeroNet, ModelConfig, masked_log_softmax


def test_forward_shape_small():
    net = AlphaZeroNet(ModelConfig(channels=32, num_blocks=2))
    x = torch.randn(4, 119, 8, 8)
    p, v = net(x)
    assert p.shape == (4, POLICY_SIZE)
    assert v.shape == (4,)
    assert torch.all(v.abs() <= 1.0 + 1e-6)


def test_forward_real_position():
    net = AlphaZeroNet(ModelConfig(channels=32, num_blocks=2))
    net.eval()
    board = chess.Board()
    x = torch.from_numpy(encode_board(board)).unsqueeze(0)
    p, v = net(x)
    assert p.shape == (1, POLICY_SIZE)
    assert v.shape == (1,)


def test_param_count_reasonable():
    """20x256 should land ~20M params. Small configs are cheaper."""
    big = AlphaZeroNet(ModelConfig(channels=256, num_blocks=20))
    n = big.num_parameters()
    assert 15_000_000 < n < 30_000_000, f"unexpected param count {n:,}"


def test_masked_log_softmax_illegal_is_neg_inf():
    logits = torch.randn(2, POLICY_SIZE)
    board = chess.Board()
    mask = torch.from_numpy(legal_move_mask(board)).unsqueeze(0).expand(2, -1)
    lp = masked_log_softmax(logits, mask)
    # Illegal entries should have negligible probability.
    assert (lp[~mask] < -50).all()
    # Legal entries sum to ~1 in prob space.
    probs = lp.exp()
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert (probs[~mask] < 1e-20).all()


def test_backward_runs():
    net = AlphaZeroNet(ModelConfig(channels=16, num_blocks=2))
    x = torch.randn(2, 119, 8, 8, requires_grad=False)
    p, v = net(x)
    loss = p.sum() + v.sum()
    loss.backward()
    # Any grad should be non-None.
    for param in net.parameters():
        assert param.grad is not None
