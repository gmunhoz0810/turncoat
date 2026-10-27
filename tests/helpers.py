"""Synthetic dinovol embeddings: one smooth random profile along the normal, shifted by `shift_vx` on chosen nodes
(a node on another sheet sees the same layered pattern displaced along its normal)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

T = np.arange(-48, 49, 2.0)


def base_profile(d: int = 16, seed: int = 0, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.arange(-200, 200, 1.0)
    raw = rng.normal(size=(n, d))
    k = np.exp(-0.5 * (np.arange(-12, 13) / 4.0) ** 2)
    sm = np.stack([np.convolve(raw[:, j], k, mode="same") for j in range(d)], 1)
    return x, sm / np.linalg.norm(sm, axis=1, keepdims=True)


def write_embeddings(path: Path, ok: np.ndarray, shift_vx: np.ndarray, d: int = 16) -> Path:
    x, prof = base_profile(d)
    rows = []
    for s in shift_vx[ok]:
        e = np.stack([np.interp(T + s, x, prof[:, j]) for j in range(d)], 1)
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        rows.append(np.round(e * 127).astype(np.int8))
    np.savez(path, ok=ok, t=T, dense=np.stack(rows))
    return Path(path)
