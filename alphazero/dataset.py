"""Torch Dataset wrappers around our shard .npz files."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset

from .encoding import NUM_PLANES, POLICY_SIZE, decode_uint8_to_float32


def convert_shard_to_npy(npz_path: str, delete_npz: bool = False) -> None:
    """Split a compressed .npz shard into three memory-mappable .npy files.

    Dramatically speeds up dataset construction (100x) because loading an .npz
    decompresses the whole file on first array access, while .npy with
    mmap_mode='r' is instant.
    """
    base = npz_path[:-4]
    out = {
        "planes": base + "_planes.npy",
        "policy_idx": base + "_policy_idx.npy",
        "values": base + "_values.npy",
    }
    if all(os.path.exists(p) for p in out.values()):
        return  # already converted
    data = np.load(npz_path)
    for key, path in out.items():
        np.save(path, data[key])
    if delete_npz:
        os.remove(npz_path)


class ShardDataset(Dataset):
    """Memory-maps one shard and serves (planes, policy, value) tuples.

    Accepts either a .npz (slow, full-decompress on access) or a triple of
    .npy files (fast, memory-mapped). If given an .npz, looks for sibling
    .npy files (same basename, _planes.npy etc) and prefers those.
    """

    def __init__(self, path: str):
        self.path = path
        base = path[:-4] if path.endswith(".npz") else path
        planes_npy = base + "_planes.npy"
        policy_npy = base + "_policy_idx.npy"
        values_npy = base + "_values.npy"
        if os.path.exists(planes_npy) and os.path.exists(policy_npy) and os.path.exists(values_npy):
            self.planes = np.load(planes_npy, mmap_mode="r")
            self.policy_idx = np.load(policy_npy, mmap_mode="r")
            self.values = np.load(values_npy, mmap_mode="r")
        else:
            data = np.load(path)
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
