from __future__ import annotations

import glob
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile


@dataclass
class NormalizationStats:
    noisy_mean: float
    noisy_std: float
    clean_mean: float
    clean_std: float


@dataclass
class DatasetBundle:
    noisy_norm: np.ndarray
    clean_norm: np.ndarray
    train_ids: list[int]
    val_ids: list[int]
    stats: NormalizationStats
    file_names: list[str]


def list_tif_files(folder: str) -> list[str]:
    return sorted(glob.glob(str(Path(folder) / "*.tif")))


def load_image_pairs(noisy_dir: str, clean_dir: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    noisy_files = list_tif_files(noisy_dir)
    clean_files = list_tif_files(clean_dir)
    if len(noisy_files) != len(clean_files) or not noisy_files:
        raise ValueError("Mismatched or empty noisy/clean image folders")

    noisy_arr: list[np.ndarray] = []
    clean_arr: list[np.ndarray] = []
    names: list[str] = []

    for nf, cf in zip(noisy_files, clean_files):
        n = np.squeeze(tifffile.imread(nf).astype(np.float32))
        c = np.squeeze(tifffile.imread(cf).astype(np.float32))
        if n.shape != c.shape:
            raise ValueError(f"Shape mismatch for {nf} and {cf}")
        noisy_arr.append(n)
        clean_arr.append(c)
        names.append(Path(nf).name)

    return np.stack(noisy_arr, axis=0), np.stack(clean_arr, axis=0), names


def split_ids(n_items: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    ids = list(range(n_items))
    random.Random(seed).shuffle(ids)
    val_count = max(1, int(n_items * val_ratio))
    val_ids = sorted(ids[:val_count])
    train_ids = sorted(ids[val_count:])
    return train_ids, val_ids


def build_dataset_bundle(
    noisy_dir: str,
    clean_dir: str,
    val_ratio: float,
    seed: int,
) -> DatasetBundle:
    noisy, clean, names = load_image_pairs(noisy_dir, clean_dir)

    nm, ns = float(noisy.mean()), float(noisy.std())
    cm, cs = float(clean.mean()), float(clean.std())

    noisy_norm = (noisy - nm) / (ns + 1e-8)
    clean_norm = (clean - cm) / (cs + 1e-8)

    train_ids, val_ids = split_ids(noisy_norm.shape[0], val_ratio, seed)
    stats = NormalizationStats(nm, ns, cm, cs)

    return DatasetBundle(noisy_norm, clean_norm, train_ids, val_ids, stats, names)
