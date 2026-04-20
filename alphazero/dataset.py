"""Torch Dataset wrappers around our shard .npz files."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset

from .encoding import NUM_PLANES, POLICY_SIZE, decode_uint8_to_float32


class ShardDataset(Dataset):
    """Memory-maps one shard .npz and serves (planes, policy, value) tuples.

    policy is materialized on-the-fly from policy_idx -> one-hot, to save disk.
    """

    def __init__(self, path: str):
        self.path = path
        data = np.load(path)
        # .npz lazy-loads each array on access; keep references.
        self.planes = data["planes"]
        self.policy_idx = data["policy_idx"]
        self.values = data["values"]
        self._len = len(self.values)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, i: int):
        raw = np.asarray(self.planes[i])
        if raw.dtype == np.uint8:
            planes = torch.from_numpy(decode_uint8_to_float32(raw))
        else:
            planes = torch.from_numpy(raw.astype(np.float32))
        policy = torch.zeros(POLICY_SIZE, dtype=torch.float32)
        policy[int(self.policy_idx[i])] = 1.0
        value = torch.tensor(float(self.values[i]), dtype=torch.float32)
        return {"planes": planes, "policy": policy, "value": value}


def build_dataset_from_dir(directory: str) -> Dataset:
    """Concatenate every shard .npz in a directory into one big dataset."""
    shards = sorted(glob.glob(os.path.join(directory, "*.npz")))
    if not shards:
        raise FileNotFoundError(f"no .npz shards in {directory}")
    return ConcatDataset([ShardDataset(s) for s in shards])


class ReplayBufferDataset(Dataset):
    """Thin adapter so the RL replay buffer looks like a Dataset."""

    def __init__(self, buffer, steps_per_epoch: int):
        self.buffer = buffer
        self.steps_per_epoch = steps_per_epoch

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __getitem__(self, i: int):
        # Sample one row at a time; collate_fn will stack.
        planes, policies, values = self.buffer.sample(1)
        return {
            "planes": torch.from_numpy(planes[0]),
            "policy": torch.from_numpy(policies[0]),
            "value": torch.tensor(values[0], dtype=torch.float32),
        }
