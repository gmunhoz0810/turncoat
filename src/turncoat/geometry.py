"""Grid geometry on tifxyz surfaces: normals, decimation, flattened (render) coordinates, areas."""
from __future__ import annotations

import warnings

import numpy as np

from .tifxyz import Surface


def normals(xyz: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit normal per vertex from central (or one-sided) differences along the grid.

    Returns (normals (H, W, 3), ok) where ok marks vertices with both tangents. The sign convention is fixed by the
    grid orientation (cross(d/dcol, d/drow)), so it is consistent across the whole surface.
    """
    p = np.where(valid[..., None], xyz, np.nan)

    def diff(axis: int) -> np.ndarray:
        f = np.full_like(p, np.nan)
        b = np.full_like(p, np.nan)
        if axis == 0:
            f[:-1] = p[1:]
            b[1:] = p[:-1]
        else:
            f[:, :-1] = p[:, 1:]
            b[:, 1:] = p[:, :-1]
        c = (f - b) / 2.0
        c = np.where(np.isnan(c), f - p, c)
        return np.where(np.isnan(c), p - b, c)

    t_row, t_col = diff(0), diff(1)
    n = np.cross(t_col, t_row)
    ln = np.linalg.norm(n, axis=-1, keepdims=True)
    ok = valid & np.isfinite(ln[..., 0]) & (ln[..., 0] > 1e-6)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = n / ln
    return np.where(ok[..., None], unit, np.nan), ok


def decimate(surface: Surface, factor: int) -> tuple[Surface, int]:
    """Keep every `factor`-th grid row/column. Returns the new surface and the factor used (>= 1)."""
    f = max(1, int(factor))
    if f == 1:
        return surface, 1
    xyz = surface.xyz[::f, ::f]
    valid = surface.valid[::f, ::f]
    scale = surface.meta.get("scale", [1.0, 1.0])
    meta = {**surface.meta, "scale": [float(scale[0]) / f, float(scale[1]) / f], "turncoat_decimated": f}
    return Surface(xyz=xyz.copy(), valid=valid.copy(), meta=meta, path=surface.path), f


def auto_decimation(surface: Surface, target_step_vx: float) -> int:
    """Integer factor that brings the grid step closest to target_step_vx without exceeding it by > 50 %."""
    step = surface.grid_step()
    return max(1, int(round(target_step_vx / max(step, 1e-6))))


def quad_area_vx2(xyz: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Area of each grid quad (H-1, W-1) in voxel^2; 0 where a corner is missing."""
    q = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
    a, b, c, d = xyz[:-1, :-1], xyz[1:, :-1], xyz[:-1, 1:], xyz[1:, 1:]
    t1 = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=-1)
    t2 = 0.5 * np.linalg.norm(np.cross(b - d, c - d), axis=-1)
    return np.where(q, np.nan_to_num(t1 + t2), 0.0)


def node_area_mm2(surface: Surface, voxel_um: float) -> float:
    """Mean surface area per grid node in mm^2 (used to report region sizes)."""
    qa = quad_area_vx2(surface.xyz, surface.valid)
    nq = max(1, int((qa > 0).sum()))
    return float(qa.sum() / nq) * (voxel_um * 1e-3) ** 2


def grid_distance(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance in grid nodes from every node to the nearest True node of mask."""
    from scipy.ndimage import distance_transform_edt
    if not mask.any():
        return np.full(mask.shape, np.inf)
    return distance_transform_edt(~mask)


def filled_normals(surface: Surface) -> np.ndarray:
    """Unit normals at every valid node; nodes without both tangents borrow the mean of their neighbours."""
    nrm, ok = normals(surface.xyz, surface.valid)
    g = np.where(ok[..., None], nrm, np.nan)
    for _ in range(3):
        miss = surface.valid & ~np.isfinite(g).all(-1)
        if not miss.any():
            break
        stack = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            stack.append(np.roll(np.roll(g, dr, 0), dc, 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmean(np.stack(stack), 0)
        ln = np.linalg.norm(m, axis=-1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = m / ln
        g = np.where(miss[..., None] & np.isfinite(m), m, g)
    return g
