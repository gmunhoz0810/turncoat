"""What every v2 channel returns, and the clean-surface CDF that puts channels on one scale."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ChannelOutput:
    """One channel on the mesh grid.

    score:     (H, W) float, larger = more switch-like, NaN where the channel has no evidence.
    offset_vx: (H, W) float or None. Signed displacement along the node normal that would put the node back on the
               sheet the rest of the mesh follows, as this channel's evidence predicts (NaN where unknown).
    meta:      small JSON-able facts (parameters, pitch, data source, timings).
    """

    name: str
    score: np.ndarray
    offset_vx: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.score.ndim != 2:
            raise ValueError(f"channel {self.name}: score must be 2-D")
        if self.offset_vx is not None and self.offset_vx.shape != self.score.shape:
            raise ValueError(f"channel {self.name}: offset_vx and score differ in shape")

    @property
    def coverage(self) -> float:
        return float(np.isfinite(self.score).mean()) if self.score.size else 0.0


@dataclass(frozen=True)
class Unavailable:
    """A channel that could not run on this input, and why (reported, never silently dropped)."""

    name: str
    reason: str


def midrank_cdf(sorted_ref: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Empirical CDF of `values` against a sorted reference sample, ties at their mid-rank, NaN kept.

    Mid-rank matters: most clean nodes of a sparse channel score exactly 0; with a right-continuous CDF they would
    map to u ~ 1 and look alarming. Here a value tied with a share q of the reference maps to the middle of that
    share, so "this channel says nothing" sits near the middle of the scale."""
    ref = np.asarray(sorted_ref, dtype=np.float64)
    if ref.ndim != 1 or ref.size == 0:
        raise ValueError("reference sample must be a non-empty 1-D array")
    v = np.asarray(values, dtype=np.float64)
    lo = np.searchsorted(ref, np.nan_to_num(v, nan=0.0), side="left")
    hi = np.searchsorted(ref, np.nan_to_num(v, nan=0.0), side="right")
    u = (lo + hi) / (2.0 * ref.size)
    return np.where(np.isfinite(v), u, np.nan)


def reference_sample(values: np.ndarray, size: int = 20001) -> np.ndarray:
    """A sorted, evenly subsampled copy of the finite values (keeps tie shares), for storing a CDF compactly."""
    v = np.sort(np.asarray(values, dtype=np.float64)[np.isfinite(values)])
    if v.size == 0:
        raise ValueError("no finite values to build a reference from")
    if v.size <= size:
        return v
    idx = np.round(np.linspace(0, v.size - 1, size)).astype(np.int64)
    return v[idx]
