"""The sheet-switch rule: which nodes of a segment sit on another sheet than the rest of it.

Signals:
- dinovol: a2 = |offset from the mesh's dominant embedding layer| / layer pitch, per node (embeddings.py);
- Turncoat's `leave` / `step` alarms: where the segment leaves the sheet instance it rides on in the organisers' surface
  prediction, or steps along its own grid (alarms/).

Rule (fitted once on the dev meshes PHerc0139 w045 + w046, 192 settings, highest pooled node-exact F1; then frozen
before any test): nodes with a2 >= A2_MIN form 8-connected regions; a region is kept when it covers >= MIN_MM2 and at
least V3_MIN of its sampled nodes lie within V3_MM of a leave / step alarm. Dev: recall 88.8 %, precision 92.5 %,
F1 0.906. Twelve held-out meshes of four scrolls, scored once: F1 0.537 (see the README).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

EIGHT = np.ones((3, 3), bool)
A2_MIN = 0.3          # layers off the dominant embedding layer
MIN_MM2 = 1.0         # smallest region kept
V3_MIN = 1            # leave / step alarms a region needs within V3_MM
V3_MM = 1.0
SAMPLE_CAP = 2000     # region nodes sampled for the alarm test
LIKELIHOOD_FULL = 0.5  # a2 at which the review likelihood reaches 1.0 (half a layer off)


def min_nodes(xyz: np.ndarray, voxel_um: float, mm2: float = MIN_MM2) -> int:
    """Grid nodes covering `mm2` on this mesh (median grid spacing in both directions)."""
    d0 = np.linalg.norm(np.diff(xyz, axis=0), axis=-1)
    d1 = np.linalg.norm(np.diff(xyz, axis=1), axis=-1)
    a = float(np.nanmedian(d0) * np.nanmedian(d1)) * (voxel_um * 1e-3) ** 2
    return max(1, int(np.ceil(mm2 / max(a, 1e-12))))


def a2_map(off: np.ndarray, pitch: float, ok: np.ndarray) -> np.ndarray:
    """(H, W) |offset| / pitch on embedded nodes, 0 elsewhere."""
    if not pitch > 0:
        raise ValueError(f"layer pitch must be positive, got {pitch}")
    return np.where(ok, np.abs(np.nan_to_num(off)) / pitch, 0.0)


def flags(off: np.ndarray, pitch: float, ok: np.ndarray, xyz: np.ndarray, valid: np.ndarray, voxel_um: float,
          alarms_xyz: np.ndarray) -> np.ndarray:
    """(H, W) bool: nodes flagged as off the segment's dominant sheet."""
    if off.shape != valid.shape or ok.shape != valid.shape or xyz.shape[:2] != valid.shape:
        raise ValueError("offset, embedded mask, xyz and valid mask must share the mesh grid")
    a2 = a2_map(off, pitch, ok)
    cand = ok & valid & (a2 >= A2_MIN)
    lab, _ = ndimage.label(cand, structure=EIGHT)
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_nodes(xyz, voxel_um, MIN_MM2)
    keep[0] = False
    out = keep[lab]
    if V3_MIN > 0:
        tree = cKDTree(alarms_xyz) if len(alarms_xyz) else None
        for k in np.nonzero(keep)[0]:
            reg = lab == k
            pts = xyz[reg]
            pts = pts[np.isfinite(pts).all(1)][:: max(1, int(reg.sum()) // SAMPLE_CAP)]
            hits = 0
            if tree is not None and len(pts):
                d, _ = tree.query(pts, distance_upper_bound=V3_MM * 1000.0 / voxel_um)
                hits = int(np.isfinite(d).sum())
            if hits < V3_MIN:
                out &= ~reg
    return out


def likelihood(off: np.ndarray, pitch: float, ok: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """(H, W) float32 review map: a2 / LIKELIHOOD_FULL clipped to 0-1 (1.0 at half a layer off); NaN where the mesh has
    no embedding."""
    a2 = a2_map(off, pitch, ok)
    return np.where(ok & valid, np.clip(a2 / LIKELIHOOD_FULL, 0.0, 1.0), np.nan).astype(np.float32)
