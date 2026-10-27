"""CT layer tracker: where does the surface's main papyrus layer lie along each vertex normal?

1. For every pair of grid neighbours, find the sub-voxel shift that best aligns their CT profiles (normalised
   cross-correlation). On one sheet the shift is ~0; where the surface slides across layers the profiles move.
2. Integrate all shifts by robust weighted least squares over the grid (iteratively reweighted, so edges that
   disagree with their loops lose their vote). The result L(v) is the position of one reference layer along v's
   normal. A surface that stays on its sheet has L ~ constant; a surface that switched sheets has L jump by a
   whole layer spacing between the two parts.
3. The reference is the dominant layer: the one carrying the largest share of the surface. offset = -(L - L0)
   is how far each vertex sits from that layer along its own normal, in voxels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

from .profiles import Profiles


class TrackerError(RuntimeError):
    """Tracking is impossible on this input (no layers, too few edges)."""


@dataclass(frozen=True)
class Edges:
    i: np.ndarray       # profile index of the first node
    j: np.ndarray       # profile index of the second node
    shift: np.ndarray   # b(k) ~ a(k + shift): layers moved by -shift from i to j
    ncc: np.ndarray     # peak NCC of the aligned profiles


@dataclass(frozen=True)
class Track:
    offset: np.ndarray      # (H, W) vx from the dominant layer along the normal, NaN where unknown
    conf: np.ndarray        # (H, W) summed edge weight per node (0 = no usable edge)
    resid: np.ndarray       # (H, W) median |loop residual| of the node's edges, vx
    spacing: float          # layer spacing (vx) measured from the profiles
    weights: np.ndarray     # final robust weight per edge
    edges: Edges


def layer_spacing(prof: np.ndarray, lo: int = 5, hi: int | None = None) -> float:
    """Layer spacing from the mean autocorrelation of the profiles: the first peak after the first trough."""
    if prof.ndim != 2 or prof.shape[1] < 2 * lo + 3:
        raise TrackerError("profiles too short to measure a layer spacing")
    pc = prof - prof.mean(1, keepdims=True)
    n = pc.shape[1]
    hi = hi or n // 2
    ac = np.array([np.mean(pc[:, : n - lag] * pc[:, lag:]) for lag in range(hi)])
    if ac[0] <= 0:
        raise TrackerError("profiles carry no intensity variation (empty or masked region?)")
    ac = ac / ac[0]
    troughs = [i for i in range(2, len(ac) - 1) if ac[i] < ac[i - 1] and ac[i] <= ac[i + 1]]
    if not troughs:
        raise TrackerError("no layering found in the CT profiles")
    peaks = [i for i in range(troughs[0] + 1, len(ac) - 1) if ac[i] > ac[i - 1] and ac[i] >= ac[i + 1]]
    if not peaks or peaks[0] < lo:
        raise TrackerError("no layering found in the CT profiles")
    lag = peaks[0]
    # parabolic refinement of the autocorrelation peak
    a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
    den = a - 2 * b + c
    frac = 0.5 * (a - c) / den if abs(den) > 1e-9 else 0.0
    return float(lag + np.clip(frac, -0.5, 0.5))


def grid_edges(pr: Profiles, diagonal: bool = False, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Profile-index pairs of grid neighbours (down, right; plus both diagonals if asked) inside mask."""
    idx = pr.index_grid()
    if mask is not None:
        idx = np.where(mask, idx, -1)
    H, W = idx.shape
    offs = [(1, 0), (0, 1)] + ([(1, 1), (1, -1)] if diagonal else [])
    ii, jj = [], []
    for dr, dc in offs:
        r0, r1 = 0, H - dr
        c0, c1 = max(0, -dc), W - max(0, dc)
        a = idx[r0:r1, c0:c1]
        b = idx[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        m = (a >= 0) & (b >= 0)
        ii.append(a[m])
        jj.append(b[m])
    return np.concatenate(ii), np.concatenate(jj)


def pair_shifts(a: np.ndarray, b: np.ndarray, W: int, maxshift: float, step: float = 0.25,
                chunk: int = 200_000) -> tuple[np.ndarray, np.ndarray]:
    """For profile pairs a, b (n, 2K+1): shift with b(k) ~ a(k + shift) over |k| <= W; returns (shift, peak NCC)."""
    K = (a.shape[1] - 1) // 2
    if W + maxshift + 1 > K:
        raise TrackerError(f"window {W} + maxshift {maxshift} exceeds profile half-width {K}")
    k = np.arange(-W, W + 1, dtype=np.float32)
    shifts = np.arange(-maxshift, maxshift + 1e-6, step, dtype=np.float32)
    out_s = np.zeros(len(a), np.float32)
    out_n = np.full(len(a), -1.0, np.float32)
    for s0 in range(0, len(a), chunk):
        A = a[s0:s0 + chunk]
        Bw = b[s0:s0 + chunk, K - W:K + W + 1]
        Bw = (Bw - Bw.mean(1, keepdims=True)) / np.maximum(Bw.std(1, keepdims=True), 1e-3)
        best = np.full(len(A), -2.0, np.float32)
        arg = np.zeros(len(A), np.float32)
        for s in shifts:
            pos = k + s + K
            i0 = np.floor(pos).astype(int)
            f = pos - i0
            Aw = A[:, i0] * (1 - f) + A[:, i0 + 1] * f
            Aw = (Aw - Aw.mean(1, keepdims=True)) / np.maximum(Aw.std(1, keepdims=True), 1e-3)
            ncc = (Aw * Bw).mean(1)
            m = ncc > best
            best[m] = ncc[m]
            arg[m] = s
        out_s[s0:s0 + chunk] = arg
        out_n[s0:s0 + chunk] = best
    return out_s, out_n


def _solve(n: int, i: np.ndarray, j: np.ndarray, d: np.ndarray, w: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    """Weighted least squares L[j] - L[i] = d with a weak L = 0 prior on every node (fixes the free constant)."""
    E = len(i)
    sw = np.sqrt(w)
    rows = np.r_[np.arange(E), np.arange(E), E + np.arange(n)]
    cols = np.r_[j, i, np.arange(n)]
    vals = np.r_[sw, -sw, np.full(n, lam)]
    A = coo_matrix((vals, (rows, cols)), shape=(E + n, n)).tocsr()
    rhs = np.r_[sw * d, np.zeros(n)]
    return lsqr(A, rhs, atol=1e-9, btol=1e-9, iter_lim=50_000)[0]


def integrate(n: int, edges: Edges, min_ncc: float, spacing: float, irls: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Robust integration of edge shifts into L. Returns (L per profile node, final edge weights)."""
    d = -edges.shift.astype(np.float64)
    w0 = np.where(edges.ncc >= min_ncc, np.clip(edges.ncc, 0, 1) ** 2, 0.0).astype(np.float64)
    if (w0 > 0).sum() < 3:
        raise TrackerError("too few neighbour pairs with matching CT profiles")
    w = w0.copy()
    L = _solve(n, edges.i, edges.j, d, w)
    c = 0.25 * spacing
    for _ in range(irls):
        r = (L[edges.j] - L[edges.i]) - d
        w = w0 / (1.0 + (r / c) ** 2)          # Cauchy weights: loop-inconsistent edges lose their vote
        L = _solve(n, edges.i, edges.j, d, w)
    return L, w


def dominant_level(L: np.ndarray, weight: np.ndarray, spacing: float) -> float:
    """Centre of the half-spacing-wide window of L that holds the most weight (the dominant layer)."""
    ok = np.isfinite(L) & (weight > 0)
    if not ok.any():
        return float(np.nanmedian(L))
    v, w = L[ok], weight[ok]
    lo, hi = np.floor(v.min()) - spacing, np.ceil(v.max()) + spacing
    bins = np.arange(lo, hi + 0.5, 0.5)
    h, _ = np.histogram(v, bins=bins, weights=w)
    win = max(1, int(round(spacing / 4 / 0.5)))          # +- a quarter spacing: one layer, never two
    cs = np.convolve(h, np.ones(2 * win + 1), mode="same")
    centre = 0.5 * (bins[:-1] + bins[1:])[int(np.argmax(cs))]
    near = np.abs(v - centre) < spacing / 3
    if not near.any():
        return float(centre)
    order = np.argsort(v[near])
    cw = np.cumsum(w[near][order])
    return float(v[near][order][np.searchsorted(cw, cw[-1] / 2)])   # weighted median of that layer


def node_stats(pr: Profiles, edges: Edges, w: np.ndarray, L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-node confidence (sum of robust edge weights) and median |loop residual| of its edges."""
    n = len(pr.rc)
    conf = np.zeros(n)
    np.add.at(conf, edges.i, w)
    np.add.at(conf, edges.j, w)
    r = np.abs((L[edges.j] - L[edges.i]) + edges.shift)
    use = w > 0
    resid = np.full(n, np.nan)
    order = np.argsort(np.r_[edges.i[use], edges.j[use]], kind="stable")
    nodes = np.r_[edges.i[use], edges.j[use]][order]
    vals = np.r_[r[use], r[use]][order]
    if len(nodes):
        starts = np.r_[0, np.flatnonzero(np.diff(nodes)) + 1]
        meds = [np.median(vals[s:e]) for s, e in zip(starts, np.r_[starts[1:], len(nodes)])]
        resid[nodes[starts]] = meds
    return conf, resid


def layered_mask(pr: Profiles, W: int = 22, min_rel: float = 0.35) -> np.ndarray:
    """(H, W) bool: nodes whose CT window shows layering. Air, masked-out CT and featureless fog fail: their
    window contrast is below `min_rel` of the surface's median contrast."""
    K = pr.K
    win = pr.prof[:, K - W:K + W + 1]
    contrast = win.std(1)
    ref = float(np.median(contrast[contrast > 0])) if (contrast > 0).any() else 0.0
    ok = contrast >= min_rel * ref if ref > 0 else np.zeros(len(contrast), bool)
    grid = np.zeros(pr.shape, bool)
    grid[pr.rc[:, 0], pr.rc[:, 1]] = ok
    return grid


def track(pr: Profiles, W: int = 22, maxshift: float = 8.0, min_ncc: float = 0.3, diagonal: bool = True,
          spacing: float | None = None, mask: np.ndarray | None = None, irls: int = 4) -> Track:
    """Run the tracker on sampled profiles. `mask` (H, W) restricts which nodes take part; nodes without
    visible layering are always left out (reported as unknown)."""
    sp = float(spacing) if spacing else layer_spacing(pr.prof)
    lay = layered_mask(pr, W)
    mask = lay if mask is None else (mask & lay)
    i, j = grid_edges(pr, diagonal=diagonal, mask=mask)
    if len(i) < 3:
        raise TrackerError("surface has fewer than 3 neighbour pairs")
    s, ncc = pair_shifts(pr.prof[i], pr.prof[j], W, maxshift)
    edges = Edges(i=i, j=j, shift=s, ncc=ncc)
    L, w = integrate(len(pr.rc), edges, min_ncc, sp, irls=irls)
    conf_n, resid_n = node_stats(pr, edges, w, L)
    used = conf_n > 0
    L = np.where(used, L, np.nan)
    L0 = dominant_level(L, conf_n, sp)
    off = -(L - L0)
    grid = np.full(pr.shape, np.nan)
    grid[pr.rc[:, 0], pr.rc[:, 1]] = off
    cg = np.zeros(pr.shape)
    cg[pr.rc[:, 0], pr.rc[:, 1]] = conf_n
    rg = np.full(pr.shape, np.nan)
    rg[pr.rc[:, 0], pr.rc[:, 1]] = resid_n
    return Track(offset=grid, conf=cg, resid=rg, spacing=sp, weights=w, edges=edges)
