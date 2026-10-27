"""Surface-prediction helpers for the `leave` channel (from Turncoat v2's instances channel)."""
from __future__ import annotations

import numpy as np


def binarise(box: np.ndarray) -> np.ndarray:
    """Foreground of a surface prediction: > 127 for 0/255 masks, > 0 for 0/1 masks."""
    return box > (127 if box.size and box.max() > 1 else 0)


def erode6(b: np.ndarray) -> np.ndarray:
    """One-voxel erosion with the 6-neighbour cross, outside = background: equal to
    scipy.ndimage.binary_erosion(b) (checked in the tests) and ~25x faster."""
    o = b.copy()
    if min(b.shape) < 3:
        return np.zeros_like(b)
    o[1:] &= b[:-1]
    o[:-1] &= b[1:]
    o[:, 1:] &= b[:, :-1]
    o[:, :-1] &= b[:, 1:]
    o[:, :, 1:] &= b[:, :, :-1]
    o[:, :, :-1] &= b[:, :, 1:]
    o[0] = o[-1] = False
    o[:, 0] = o[:, -1] = False
    o[:, :, 0] = o[:, :, -1] = False
    return o
