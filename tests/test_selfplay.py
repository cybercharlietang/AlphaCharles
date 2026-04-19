"""Tests for self-play and replay buffer."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from alphazero.encoding import NUM_PLANES, POLICY_SIZE
from alphazero.mcts import MCTSConfig
from alphazero.model import AlphaZeroNet, ModelConfig
from alphazero.replay import ReplayBuffer
from alphazero.selfplay import SelfPlayConfig, play_game


@pytest.fixture(scope="module")
def tiny_net():
    torch.manual_seed(0)
    net = AlphaZeroNet(ModelConfig(channels=16, num_blocks=2))
    net.eval()
    return net


def test_play_short_game(tiny_net):
    """Run a very short game with a 16-sim MCTS and 40-ply limit."""
    cfg = SelfPlayConfig(
        mcts=MCTSConfig(num_simulations=16, add_root_noise=True, temperature_moves=4),
        move_limit=40,
        resign_threshold=None,
    )
    rec = play_game(tiny_net, torch.device("cpu"), cfg)
    assert rec.ply_count > 0
    assert rec.ply_count <= 40
    assert rec.planes.shape == (rec.ply_count, NUM_PLANES, 8, 8)
    assert rec.policies.shape == (rec.ply_count, POLICY_SIZE)
    assert rec.values.shape == (rec.ply_count,)
    # Every value in {-1, 0, 1}.
    assert set(np.unique(rec.values).tolist()) <= {-1.0, 0.0, 1.0}
    # Policy rows sum to 1.
    sums = rec.policies.sum(axis=1)
    assert np.allclose(sums, np.ones_like(sums), atol=1e-5)


def test_value_alternates_for_decisive_result(tiny_net):
    """If a game ends decisively, value targets alternate in sign ply by ply
    (white's mover value = +outcome, black's = -outcome)."""
    cfg = SelfPlayConfig(
        mcts=MCTSConfig(num_simulations=8, add_root_noise=True, temperature_moves=2),
        move_limit=20,
        resign_threshold=None,
    )
    np.random.seed(0)
    torch.manual_seed(0)
    # Run a few games until we get a decisive one; move_limit=20 usually draws,
    # so this test may skip if it doesn't happen. Just check alternation when decisive.
    for _ in range(5):
        rec = play_game(tiny_net, torch.device("cpu"), cfg)
        if rec.result == "1/2-1/2":
            continue
        diffs = np.diff(rec.values)
        assert np.all(diffs != 0), "values should flip each ply in a decisive game"
        break


def test_replay_buffer_wraparound():
    buf = ReplayBuffer(capacity=10)
    planes = np.zeros((7, NUM_PLANES, 8, 8), dtype=np.float32)
    policies = np.zeros((7, POLICY_SIZE), dtype=np.float32)
    values = np.arange(7, dtype=np.float32)
    buf.add(planes, policies, values)
    assert len(buf) == 7
    # Add 6 more -> should wrap.
    values2 = np.arange(10, 16, dtype=np.float32)
    buf.add(np.zeros((6, NUM_PLANES, 8, 8), dtype=np.float32),
            np.zeros((6, POLICY_SIZE), dtype=np.float32), values2)
    assert len(buf) == 10
    # Buffer now holds values: positions 0..6 initially; wrap wrote new at 7,8,9,0,1,2.
    # So buf.values should be [13,14,15,3,4,5,6,10,11,12].
    assert set(buf.values.tolist()) == {3.0, 4.0, 5.0, 6.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0}


def test_replay_buffer_sample_shapes():
    buf = ReplayBuffer(capacity=100)
    buf.add(np.zeros((50, NUM_PLANES, 8, 8), dtype=np.float32),
            np.zeros((50, POLICY_SIZE), dtype=np.float32),
            np.zeros(50, dtype=np.float32))
    p, pi, v = buf.sample(8)
    assert p.shape == (8, NUM_PLANES, 8, 8)
    assert pi.shape == (8, POLICY_SIZE)
    assert v.shape == (8,)
