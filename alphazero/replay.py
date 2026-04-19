"""Replay buffer backed by memory-mapped numpy arrays.

A self-play run at ~50k games x ~100 plies = 5M samples. Each sample is:
  planes: 119*8*8*4 bytes = 30.4 KB
  policy: 4672*4    bytes = 18.7 KB
  value : 4         bytes
  -> ~49 KB per sample, ~250 GB at 5M samples.

Too big for RAM. We use three mmap'd files on disk (planes.npy, policies.npy,
values.npy) as a ring buffer with a capacity we choose up front. Writes are
O(1); sampling reads random indices.

A simpler in-memory variant is used for small experiments (SL warmstart often
fits in RAM).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .encoding import NUM_PLANES, POLICY_SIZE


@dataclass
class ReplayBuffer:
    """In-memory replay buffer (ring). Use MmapReplayBuffer for large-scale."""
    capacity: int

    def __post_init__(self):
        self.planes = np.zeros((self.capacity, NUM_PLANES, 8, 8), dtype=np.float32)
        self.policies = np.zeros((self.capacity, POLICY_SIZE), dtype=np.float32)
        self.values = np.zeros(self.capacity, dtype=np.float32)
        self._idx = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, planes: np.ndarray, policies: np.ndarray, values: np.ndarray) -> None:
        n = len(planes)
        if n == 0:
            return
        # Wrap-around insert.
        end = self._idx + n
        if end <= self.capacity:
            sl = slice(self._idx, end)
            self.planes[sl] = planes
            self.policies[sl] = policies
            self.values[sl] = values
        else:
            first = self.capacity - self._idx
            self.planes[self._idx:] = planes[:first]
            self.policies[self._idx:] = policies[:first]
            self.values[self._idx:] = values[:first]
            rem = n - first
            self.planes[:rem] = planes[first:]
            self.policies[:rem] = policies[first:]
            self.values[:rem] = values[first:]
        self._idx = (self._idx + n) % self.capacity
        self._size = min(self._size + n, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng()
        idx = rng.integers(0, self._size, size=batch_size)
        return self.planes[idx], self.policies[idx], self.values[idx]


class MmapReplayBuffer:
    """Disk-backed version. Use for multi-GB buffers."""

    def __init__(self, directory: str, capacity: int, create: bool = True):
        os.makedirs(directory, exist_ok=True)
        self.capacity = capacity
        self.directory = directory
        mode = "w+" if create else "r+"
        self.planes = np.memmap(os.path.join(directory, "planes.npy"), mode=mode,
                                dtype=np.float32, shape=(capacity, NUM_PLANES, 8, 8))
        self.policies = np.memmap(os.path.join(directory, "policies.npy"), mode=mode,
                                  dtype=np.float32, shape=(capacity, POLICY_SIZE))
        self.values = np.memmap(os.path.join(directory, "values.npy"), mode=mode,
                                dtype=np.float32, shape=(capacity,))
        self._idx = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, planes: np.ndarray, policies: np.ndarray, values: np.ndarray) -> None:
        n = len(planes)
        if n == 0:
            return
        end = self._idx + n
        if end <= self.capacity:
            sl = slice(self._idx, end)
            self.planes[sl] = planes
            self.policies[sl] = policies
            self.values[sl] = values
        else:
            first = self.capacity - self._idx
            self.planes[self._idx:] = planes[:first]
            self.policies[self._idx:] = policies[:first]
            self.values[self._idx:] = values[:first]
            rem = n - first
            self.planes[:rem] = planes[first:]
            self.policies[:rem] = policies[first:]
            self.values[:rem] = values[first:]
        self._idx = (self._idx + n) % self.capacity
        self._size = min(self._size + n, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng()
        idx = rng.integers(0, self._size, size=batch_size)
        return (np.asarray(self.planes[idx]),
                np.asarray(self.policies[idx]),
                np.asarray(self.values[idx]))
