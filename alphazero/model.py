"""AlphaZero ResNet.

Architecture (Silver et al. 2017, §Neural network architecture):

    input (B, 119, 8, 8)
        |
        v
    stem: 3x3 conv -> BN -> ReLU                 shape: (B, C, 8, 8)
        |
        v
    N x residual block (two 3x3 convs + skip)    shape: (B, C, 8, 8)
        |
        +------------------+
        v                  v
    policy head        value head
    (B, 4672)          (B, 1)  in [-1, 1]

Defaults: C=256 channels, N=20 blocks (~20M params). The paper used C=256, N=19;
we use 20 for a round number. Both heads follow the AZ recipe: a small
"bottleneck" conv to 2 (policy) or 1 (value) channels, flatten, then a linear layer.

Shapes are annotated on every op so you can read this like a wiring diagram.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import NUM_PLANES, POLICY_SIZE


@dataclass
class ModelConfig:
    in_planes: int = NUM_PLANES      # 119
    channels: int = 256
    num_blocks: int = 20
    policy_size: int = POLICY_SIZE   # 4672
    value_hidden: int = 256


class ResidualBlock(nn.Module):
    """Standard pre-AZ ResNet block: two 3x3 convs, BN, ReLU, skip.

    Why BN? Internal covariate shift + regularization; trains much faster on
    self-play data which has very non-stationary distribution. At inference,
    BN uses running stats (free at eval time).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, 8, 8)
        y = F.relu(self.bn1(self.conv1(x)))          # (B, C, 8, 8)
        y = self.bn2(self.conv2(y))                  # (B, C, 8, 8)
        return F.relu(x + y)                         # skip connection


class PolicyHead(nn.Module):
    """Projects the tower output to 4672 move logits.

    AZ uses a 1x1 conv to 73 channels, then the output at each of 64 squares is
    interpreted as the 73 move-types originating from that square. That's more
    parameter-efficient than a flat linear, but a flat linear is just as good
    here. We use the flat-linear variant because it matches our 4672-dim encoding
    cleanly and makes masking illegal moves trivial.
    """

    def __init__(self, channels: int, policy_size: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)  # (B, 2, 8, 8)
        self.bn = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2 * 8 * 8, policy_size)                    # -> (B, 4672)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn(self.conv(x)))            # (B, 2, 8, 8)
        y = y.flatten(1)                             # (B, 128)
        return self.fc(y)                            # (B, 4672) logits


class ValueHead(nn.Module):
    """Projects the tower output to a scalar in [-1, 1]."""

    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)  # (B, 1, 8, 8)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(8 * 8, hidden)                            # -> (B, hidden)
        self.fc2 = nn.Linear(hidden, 1)                                # -> (B, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn(self.conv(x)))            # (B, 1, 8, 8)
        y = y.flatten(1)                             # (B, 64)
        y = F.relu(self.fc1(y))                      # (B, hidden)
        return torch.tanh(self.fc2(y)).squeeze(-1)   # (B,) in [-1, 1]


class AlphaZeroNet(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg
        # Stem: map 119 input planes -> C channels.
        self.stem_conv = nn.Conv2d(cfg.in_planes, cfg.channels, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(cfg.channels)
        # Tower: N residual blocks.
        self.tower = nn.ModuleList([ResidualBlock(cfg.channels) for _ in range(cfg.num_blocks)])
        # Heads.
        self.policy_head = PolicyHead(cfg.channels, cfg.policy_size)
        self.value_head = ValueHead(cfg.channels, cfg.value_hidden)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, 119, 8, 8) -> (policy_logits (B, 4672), value (B,))"""
        y = F.relu(self.stem_bn(self.stem_conv(x)))  # (B, C, 8, 8)
        for block in self.tower:
            y = block(y)                             # (B, C, 8, 8)
        return self.policy_head(y), self.value_head(y)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """log_softmax over policy logits, setting illegal moves to -inf first.

    logits: (B, 4672)
    mask:   (B, 4672) bool, True=legal
    returns (B, 4672) log-probabilities (illegal positions are -inf).
    """
    neg_inf = torch.finfo(logits.dtype).min
    masked = logits.masked_fill(~mask, neg_inf)
    return F.log_softmax(masked, dim=-1)
