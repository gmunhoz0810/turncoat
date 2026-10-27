"""Channel A (v3): the trace LEAVES its surface-prediction instance while that instance continues alongside.

Port of the natural-switch autopsy's scan rule (turncoat-research/autopsy_natural/src/scan.py, 2026-10-25), which
replaced v2's 12-node majority-instance score (0 natural hits at its own threshold).

Per overlapping tile of TILE x TILE grid vertices (stride STRIDE): the binary prediction in the box around the tile's
vertices +- REACH vx along their normals, eroded by one voxel, 6-connected components. Per vertex: the component
under the trace (search +- SEARCH vx) and the components met along its normal at 5 <= |t| <= REACH ("alongside").

Records (grid edge v -> w, 1 <= |v - w| <= 3 along a row or column, nearest w only):
  leave      the trace is on component a at v and the vertex before it, on b != a at w and the vertex after it
  alongside  ... and a is still met alongside w's normal, or b alongside v's (the trace moved between two parallel
             instances; a hole that splits ONE sheet does not qualify)
An alarm is a leave + alongside record with at least `support` such records within 0.5 mm (rows, columns, gaps and
overlapping tiles all re-detect a real change; isolated records are label noise).

Per vertex the scan also returns two review-risk flags (never alarms): `merge` (the trace's own instance is met
alongside: it touches / fuses with another wrap there) and `hole` (no prediction under the trace). And, for the fix,
the offset along w's normal to the nearest voxel of the instance a it left (where its sheet continues).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label

from ..geometry import normals
from ..tifxyz import Surface
from .instances import binarise, erode6
from ..volume import Volume

TILE, STRIDE = 10, 7
REACH = 32.0
SEARCH = 2.5
MAX_D = 3
SIDE_MIN = 5.0
MAX_VOX = 60_000_000


@dataclass(frozen=True)
class LeaveScan:
    records: np.ndarray       # (n, 10): x y z r1 c1 r2 c2 d alongside(0/1) offset_vx_to_a_at_w (NaN if not met)
    merge: np.ndarray         # (H, W) share of the tiles seeing the vertex in which its instance is met alongside
    hole: np.ndarray          # (H, W) share of the tiles in which no instance lies under the vertex
    tiles_skipped: int
    offset: np.ndarray | None = None   # (H, W) for vertices on an instance b that a leave record entered: offset along
                                       # the vertex's own normal to the nearest voxel of the instance a it left

    def alarms(self, support: int, radius_vx: float) -> np.ndarray:
        """(m, 10) leave + alongside records with >= support such records within radius_vx (itself included)."""
        al = self.records[self.records[:, 8] > 0] if len(self.records) else self.records
        if len(al) == 0:
            return al
        from scipy.spatial import cKDTree
        cnt = np.array([len(x) for x in cKDTree(al[:, :3]).query_ball_point(al[:, :3], radius_vx)])
        return al[cnt >= support]


def tiles(H: int, W: int) -> list[tuple[int, int]]:
    rs = list(range(0, max(1, H - TILE + 1), STRIDE)) + ([H - TILE] if H > TILE else [])
    cs = list(range(0, max(1, W - TILE + 1), STRIDE)) + ([W - TILE] if W > TILE else [])
    return sorted({(max(0, r), max(0, c)) for r in rs for c in cs})


def crop_labels(vol: Volume, scale: float, pts: np.ndarray, margin: float = 3.0):
    p = pts[np.isfinite(pts).all(1)] * scale
    lo = np.maximum(np.floor(p.min(0) - margin * scale).astype(int), 0)[::-1]
    hi = np.ceil(p.max(0) + margin * scale).astype(int)[::-1] + 1
    if np.prod(hi - lo) > MAX_VOX:
        return None
    lab, n = label(erode6(binarise(vol.read_box(lo, hi))))
    return lab, lo, n


def labels_at(lp, scale: float, P: np.ndarray, N: np.ndarray, search: float = SEARCH) -> np.ndarray:
    lab, lo, _ = lp
    out = np.zeros(len(P), np.int64)
    ok = np.isfinite(P).all(1) & np.isfinite(N).all(1)
    shape = np.array(lab.shape)
    steps = np.arange(0, search + 1e-9, 0.5 / scale)
    for tt in [0.0] + [s * k for k in steps[1:] for s in (1, -1)]:
        q = np.rint((P + tt * N) * scale)[:, ::-1] - lo
        q = np.nan_to_num(q, nan=-1).astype(np.int64)
        inb = ok & np.all((q >= 0) & (q < shape), axis=1)
        need = (out == 0) & inb
        if need.any():
            out[need] = lab[q[need, 0], q[need, 1], q[need, 2]]
    return out


def scan(surface: Surface, vol: Volume, scale: float = 1.0) -> LeaveScan:
    """Scan a whole mesh. `scale` = prediction voxels per mesh voxel (1 when the prediction is in the mesh frame)."""
    P = surface.xyz
    N, nok = normals(P, surface.valid)
    H, W = surface.shape
    ok = surface.valid & nok
    recs = []
    n_tiles = np.zeros((H, W))
    n_merge = np.zeros((H, W))
    n_hole = np.zeros((H, W))
    off_map = np.full((H, W), np.nan)
    tv = np.arange(-REACH, REACH + 1e-9, 1.0)
    side = np.abs(tv) >= SIDE_MIN
    skipped = 0
    for r0, c0 in tiles(H, W):
        sl = (slice(r0, r0 + TILE), slice(c0, c0 + TILE))
        m = ok[sl]
        if m.sum() < 12:
            continue
        Xt, Nt = P[sl], N[sl]
        lp = crop_labels(vol, scale, np.concatenate([Xt[m] - REACH * Nt[m], Xt[m] + REACH * Nt[m]]))
        if lp is None:
            skipped += 1
            continue
        lab, lo, nl = lp
        if nl == 0:
            continue
        h, w = Xt.shape[:2]
        fX, fN = Xt.reshape(-1, 3), Nt.reshape(-1, 3)
        mm = m.ravel()
        lab0 = np.zeros(h * w, np.int64)
        lab0[mm] = labels_at(lp, scale, fX[mm], fN[mm])
        G = fX[mm][:, None, :] + tv[None, :, None] * fN[mm][:, None, :]
        q = (np.rint(G * scale)[..., ::-1] - lo).astype(np.int64)
        inb = np.all((q >= 0) & (q < np.array(lab.shape)), axis=-1)
        prof = np.zeros(q.shape[:2], np.int64)
        prof[inb] = lab[q[inb][:, 0], q[inb][:, 1], q[inb][:, 2]]
        rows = np.full((h * w, len(tv)), 0, np.int64)
        rows[mm] = prof
        along = [set(np.unique(r[side & (r > 0)]).tolist()) for r in rows]
        L = lab0.reshape(h, w)
        for i in range(h):
            for j in range(w):
                if not m[i, j]:
                    continue
                n_tiles[r0 + i, c0 + j] += 1
                a = L[i, j]
                if a == 0:
                    n_hole[r0 + i, c0 + j] += 1
                elif a in along[i * w + j]:
                    n_merge[r0 + i, c0 + j] += 1
        pairs: set = set()
        for di, dj in ((0, 1), (1, 0)):
            for i in range(h):
                for j in range(w):
                    for d in range(1, MAX_D + 1):
                        i2, j2 = i + di * d, j + dj * d
                        ip, jp, i3, j3 = i - di, j - dj, i2 + di, j2 + dj
                        if not (0 <= ip and 0 <= jp and i3 < h and j3 < w):
                            continue
                        a, b = L[i, j], L[i2, j2]
                        if a == 0 or b == 0 or a == b or L[ip, jp] != a or L[i3, j3] != b:
                            continue
                        al = (a in along[i2 * w + j2]) or (b in along[i * w + j])
                        hit = np.nonzero(rows[i2 * w + j2] == a)[0]
                        off = float(tv[hit[np.argmin(np.abs(tv[hit]))]]) if len(hit) else np.nan
                        x = (Xt[i, j] + Xt[i2, j2]) / 2
                        recs.append((x[0], x[1], x[2], r0 + i, c0 + j, r0 + i2, c0 + j2, d, int(al), off))
                        if al:
                            pairs.add((a, b))
                        break
        for a, b in pairs:                      # every vertex the trace keeps on b: where its sheet a continues
            for k in np.nonzero((L.ravel() == b) & mm)[0]:
                hit = np.nonzero(rows[k] == a)[0]
                if len(hit):
                    o = float(tv[hit[np.argmin(np.abs(tv[hit]))]])
                    rr, cc = r0 + k // w, c0 + k % w
                    if not np.isfinite(off_map[rr, cc]) or abs(o) < abs(off_map[rr, cc]):
                        off_map[rr, cc] = o
        del lp, lab
    with np.errstate(invalid="ignore", divide="ignore"):
        merge = np.where(n_tiles > 0, n_merge / n_tiles, np.nan)
        hole = np.where(n_tiles > 0, n_hole / n_tiles, np.nan)
    return LeaveScan(np.array(recs, float).reshape(-1, 10), merge, hole, skipped, off_map)


def offsets_from_records(recs: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """(H, W) offset (vx along the normal) to the instance the trace left, at each record's landing vertex w."""
    out = np.full(shape, np.nan)
    for r in recs:
        if np.isfinite(r[9]):
            out[int(r[5]), int(r[6])] = r[9]
    return out
