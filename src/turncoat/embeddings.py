"""dinovol embeddings on the mesh grid and the offset field the detector reads (array math only, no model here).

The embeddings are the dense head of the public `scrollprize/dinovol_8um_ps192_102126` model, sampled along every node
normal on a GPU (`turncoat embed`, see turncoat/gpu). Turncoat's robust least-squares tracker then finds, per node, the
offset along the normal from the mesh's dominant embedding layer: a node that sits on another sheet than most of the
mesh shows an offset near a whole layer.

Input file (.npz): `ok` (H, W) bool on the mesh grid, `t` (nt,) offsets along the normal (vx), `dense`
(ok.sum(), nt, 64) int8 unit vectors x 127, optional `crop` (r0, r1, c0, c1).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .tracker import Edges, TrackerError, dominant_level, integrate

PITCH_VX = 20.0     # fallback layer pitch of the embedding (first self-similarity minimum)
LOAD_BLOCK = 8192   # embedding rows converted per block


class EmbeddingError(ValueError):
    pass


@dataclass(frozen=True)
class EmbField:
    E: np.ndarray        # (H, W, nt, d) float32 unit vectors (zeros where not ok)
    ok: np.ndarray       # (H, W)
    t0: float
    dt: float

    def at(self, r: np.ndarray, c: np.ndarray, t: np.ndarray) -> np.ndarray:
        """(n, m, d) embeddings at nodes (r, c) and offsets t (n, m), linear in t, zero outside."""
        u = (t - self.t0) / self.dt
        inside = (u >= 0) & (u <= self.E.shape[2] - 1)
        u = np.clip(u, 0, self.E.shape[2] - 1)
        k0 = np.minimum(np.floor(u).astype(np.int64), self.E.shape[2] - 2)
        f = (u - k0)[..., None].astype(np.float32)
        a = self.E[r[:, None], c[:, None], k0]
        b = self.E[r[:, None], c[:, None], k0 + 1]
        v = (a * (1 - f) + b * f) * inside[..., None]
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        return np.where(n > 0, v / np.maximum(n, 1e-9), 0.0)


def load_embeddings(path: str | Path, shape: tuple[int, int], crop: tuple[int, int, int, int] | None = None
                    ) -> EmbField:
    z = np.load(path)
    for key in ("ok", "t", "dense"):
        if key not in z.files:
            raise EmbeddingError(f"{path}: missing '{key}' (expected the dinovol study's npz format)")
    ok, t, q = z["ok"].astype(bool), z["t"].astype(np.float64), z["dense"]
    if q.shape[0] != ok.sum():
        raise EmbeddingError(f"{path}: {q.shape[0]} embedding rows for {int(ok.sum())} ok nodes")
    if ok.shape != shape:
        raise EmbeddingError(f"{path}: embeddings are on a {ok.shape} grid, the checked grid is {shape} "
                         f"(file crop {z['crop'].tolist() if 'crop' in z.files else 'n/a'}, check crop {crop})")
    E = np.zeros((*ok.shape, q.shape[1], q.shape[2]), np.float32)
    rr, cc = np.nonzero(ok)                      # same order as the rows of `dense`
    for s in range(0, len(rr), LOAD_BLOCK):      # in blocks: peak memory ~ one float32 copy of the embeddings
        blk = q[s:s + LOAD_BLOCK].astype(np.float32) / 127.0
        n = np.linalg.norm(blk, axis=-1, keepdims=True)
        E[rr[s:s + LOAD_BLOCK], cc[s:s + LOAD_BLOCK]] = np.where(n > 0, blk / np.maximum(n, 1e-9), 0.0)
    return EmbField(E=E, ok=ok, t0=float(t[0]), dt=float(t[1] - t[0]))


def a2_track(f: EmbField, pitch: float = PITCH_VX, W: float = 16.0, maxshift: float = 8.0, step: float = 0.5,
             min_match: float = 0.3) -> np.ndarray:
    """(H, W) offset (vx) from the dominant embedding layer: per grid edge (4 + diagonal) the shift maximising the
    mean cosine of the two embedding profiles over |t| <= W, integrated by Turncoat's robust least squares."""
    H, Wd = f.ok.shape
    idx = -np.ones((H, Wd), np.int64)
    rr, cc = np.nonzero(f.ok)
    idx[rr, cc] = np.arange(len(rr))
    ii, jj = [], []
    for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
        r1, c1 = rr + dr, cc + dc
        inb = (r1 < H) & (c1 >= 0) & (c1 < Wd)
        a, b = idx[rr[inb], cc[inb]], idx[r1[inb], c1[inb]]
        ii.append(a[b >= 0])
        jj.append(b[b >= 0])
    i, j = np.concatenate(ii), np.concatenate(jj)
    t = np.arange(-W, W + 1e-6, f.dt)
    best = np.full(len(i), -2.0)
    arg = np.zeros(len(i))
    for ch in np.array_split(np.arange(len(i)), max(1, len(i) // 8000)):
        ri, ci, rj, cj = rr[i[ch]], cc[i[ch]], rr[j[ch]], cc[j[ch]]
        Ea = f.at(ri, ci, np.broadcast_to(t, (len(ch), len(t))))
        va = np.linalg.norm(Ea, axis=-1) > 0
        for sft in np.arange(-maxshift, maxshift + 1e-6, step):
            Eb = f.at(rj, cj, np.broadcast_to(t + sft, (len(ch), len(t))))
            v = va & (np.linalg.norm(Eb, axis=-1) > 0)
            m = (Ea * Eb).sum(-1).sum(-1) / np.maximum(v.sum(-1), 1)
            m = np.where(v.sum(-1) >= len(t) // 2, m, -2.0)
            better = m > best[ch]
            best[ch] = np.where(better, m, best[ch])
            arg[ch] = np.where(better, sft, arg[ch])
    edges = Edges(i=i, j=j, shift=arg.astype(np.float32), ncc=best.astype(np.float32))
    L, w = integrate(len(rr), edges, min_match, pitch)
    conf = np.zeros(len(rr))
    np.add.at(conf, i, w)
    np.add.at(conf, j, w)
    L = np.where(conf > 0, L, np.nan)
    off = np.full((H, Wd), np.nan)
    off[rr, cc] = (L - dominant_level(L, conf, pitch))   # sign flipped vs the study: matches the CT tracker
    return off


def sheet_pitch(f, max_t: float = 30.0, n_nodes: int = 4000, seed: int = 0) -> float:
    """Layer pitch (vx) from the embeddings: the offset where the mean cos(E(0), E(+-t)) is lowest (parabolic refine)."""
    rng = np.random.default_rng(seed)
    r, c = np.nonzero(f.ok)
    if not len(r):
        raise EmbeddingError("no embedded nodes")
    pick = rng.choice(len(r), min(n_nodes, len(r)), replace=False)
    r, c = r[pick], c[pick]
    ts = np.arange(0, max_t + 1e-6, f.dt)
    e0 = f.at(r, c, np.zeros((len(r), 1)))[:, 0]
    sims = []
    for t in ts:
        sp = (f.at(r, c, np.full((len(r), 1), t))[:, 0] * e0).sum(-1)
        sm = (f.at(r, c, np.full((len(r), 1), -t))[:, 0] * e0).sum(-1)
        sims.append(0.5 * (sp.mean() + sm.mean()))
    s = np.asarray(sims)
    k = int(np.argmin(s))
    if 0 < k < len(s) - 1:
        y0, y1, y2 = s[k - 1], s[k], s[k + 1]
        den = y0 - 2 * y1 + y2
        if den > 0:
            return float(ts[k] + 0.5 * (y0 - y2) / den * f.dt)
    return float(ts[k])


def dinovol_offset(path: str | Path, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, float]:
    """(signed offset grid in vx from the dominant embedding layer, embedded-node mask, pitch vx)."""
    try:
        f = load_embeddings(path, shape)
        pitch = sheet_pitch(f)
        off = a2_track(f, pitch)
    except (OSError, ValueError, TrackerError) as exc:
        raise EmbeddingError(f"dinovol embeddings {path}: {exc}") from exc
    return off, f.ok, pitch
