"""dinovol sheet-identity channel: dinovol embeddings sampled at arbitrary CT points (GPU only; from the Turncoat
dinovol study, turncoat-research/dinovol_test, same authors, MIT; the model is the organisers' scrollprize/dinovol_8um_ps192_102126
loaded with villa's dinovol_2 package).

The volume is tiled by a fixed lattice of WIN^3 windows (192^3, the checkpoint's training crop) (stride STRIDE, so
neighbouring windows overlap by WIN - STRIDE voxels). Every window whose core cell holds a requested point is embedded
once; each point takes the blend of every such window that covers it, weighted by a ramp that falls toward the window
edge (the same blend the villa napari viewer uses). Because the lattice is fixed in volume coordinates, the embedding
at a point does not depend on which mesh asked for it: a mesh node, its +-1 layer neighbours and a second mesh on the
same sheet all read one field.

Two embeddings per point:
  patch  backbone patch tokens (8^3 vx), trilinear at the point, L2-normalised per window, projected onto the top
         PCA axes of the uncentred token second-moment matrix (dot products preserved up to the dropped energy)
  dense  the PointDenseHead "obj" readout (64-d), trained with the sheet/fibre object-instance loss

Intensity normalisation: "global" = one robust (median / MAD after 1-99 % clipping) transform for every window
(consistent across the volume), or "window" = per-window robust, as in training.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from dinovol_2.eval.napari_visualizer import _configure_backbone_for_spatial_shape, load_backbone_from_checkpoint
from dinovol_2.dataset.normalization import normalize_robust
from dinovol_2.ops.point_embeddings import sample_grid_points

WIN = 192          # the checkpoint's global crop size (ps192): windows match what the model saw in training
STRIDE = 144
MARGIN = (WIN - STRIDE) // 2


@dataclass(frozen=True)
class NormStats:
    lo: float
    hi: float
    median: float
    scale: float

    def apply(self, block: np.ndarray) -> np.ndarray:
        x = np.clip(block.astype(np.float32), self.lo, self.hi)
        return (x - self.median) / self.scale


def global_norm_stats(blocks: list[np.ndarray]) -> NormStats:
    """Robust stats over voxels pooled from many windows (same recipe as normalize_robust)."""
    if not blocks:
        raise ValueError("need at least one block for global normalisation stats")
    v = np.concatenate([b.ravel()[:: max(1, b.size // 200_000)] for b in blocks]).astype(np.float32)
    lo, hi = np.percentile(v, [1.0, 99.0])
    v = np.clip(v, lo, hi)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826
    if not math.isfinite(mad) or mad < 1e-6:
        raise ValueError(f"degenerate intensity statistics (MAD {mad})")
    return NormStats(lo=float(lo), hi=float(hi), median=med, scale=mad)


def window_start(k: np.ndarray) -> np.ndarray:
    return k * STRIDE - MARGIN


def core_cell(zyx: np.ndarray) -> np.ndarray:
    return np.floor(zyx / STRIDE).astype(np.int64)


def ramp_weight(local: np.ndarray, overlap: int = WIN - STRIDE) -> np.ndarray:
    """Per-point blend weight inside a window: min over axes of the edge-distance ramp (1 in the core)."""
    edge = np.minimum(local + 1.0, WIN - local)
    return np.clip(edge / (overlap + 1.0), 0.0, 1.0).prod(axis=-1)


def _cell_key(k: np.ndarray) -> np.ndarray:
    """Pack lattice cell indices (non-negative, < 2^20) into one int64 per cell."""
    if (k < 0).any() or (k >= 1 << 20).any():
        raise ValueError("lattice cell index out of range")
    return (k[..., 0] << 40) | (k[..., 1] << 20) | k[..., 2]


def assign_points(zyx: np.ndarray, cores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(window id, point id) pairs, sorted by window: every core window whose full extent contains the point."""
    keys = _cell_key(cores)
    order = np.argsort(keys)
    skeys = keys[order]
    base = core_cell(zyx)
    wins, pts = [], []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                k = base + np.array([dz, dy, dx])
                local = zyx - window_start(k)
                inside = ((local >= 0) & (local <= WIN - 1)).all(1)
                idx = np.nonzero(inside)[0]
                kk = _cell_key(k[idx])
                j = np.clip(np.searchsorted(skeys, kk), 0, len(skeys) - 1)
                hit = skeys[j] == kk
                wins.append(order[j[hit]])
                pts.append(idx[hit])
    w = np.concatenate(wins)
    p = np.concatenate(pts)
    srt = np.argsort(w, kind="stable")
    return w[srt], p[srt]


class DinovolSampler:
    """Loads the checkpoint once; embeds WIN^3 blocks and reads both embeddings at points inside them."""

    def __init__(self, ckpt: str, device: str = "cuda", amp: bool = True):
        self.lb = load_backbone_from_checkpoint(ckpt, device=torch.device(device), preferred_branch="teacher")
        self.device = torch.device(device)
        self.amp = amp
        _configure_backbone_for_spatial_shape(self.lb.backbone, (WIN, WIN, WIN))
        self.has_dense = self.lb.dense_head is not None
        self.pca: torch.Tensor | None = None
        self.pca_energy: float | None = None

    def describe(self) -> dict:
        return {"branch": self.lb.source_branch, "patch_size": self.lb.patch_size, "embedding_type": self.lb.embedding_type,
                "embed_dim": int(self.lb.backbone.embed_dim), "has_dense_head": self.has_dense,
                "dense_dim": int(self.lb.dense_head.out_dim) if self.has_dense else None,
                "normalization_scheme": self.lb.normalization_scheme, "input_channels": self.lb.input_channels}

    @torch.inference_mode()
    def forward(self, blocks: np.ndarray) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """blocks (B, WIN, WIN, WIN) normalised float32 -> token grid (B, g, g, g, C) and CNN skips."""
        x = torch.from_numpy(blocks).unsqueeze(1).to(self.device)
        bb = self.lb.backbone
        bb.return_embed_skips = self.has_dense
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp and self.device.type == "cuda"):
            out = bb.forward_features(x, masks=None, view_kind="global")
        g = WIN // self.lb.patch_size[0]
        tok = out["x_norm_patchtokens"].reshape(x.shape[0], g, g, g, -1)
        return tok, out.get("embed_skips", [])

    def fit_pca(self, token_grids: list[torch.Tensor], dim: int) -> None:
        t = torch.cat([F.normalize(g.reshape(-1, g.shape[-1]).float(), dim=-1) for g in token_grids])
        m = t.T @ t / len(t)
        evals, evecs = torch.linalg.eigh(m)
        idx = torch.argsort(evals, descending=True)
        evals, evecs = evals[idx], evecs[:, idx]
        self.pca = evecs[:, :dim].contiguous()
        self.pca_energy = float(evals[:dim].sum() / evals.sum())

    @torch.inference_mode()
    def point_features(self, tok: torch.Tensor, skips: list[torch.Tensor], b: int,
                       local_zyx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Embeddings at points (P, 3) in window-local voxel coords of batch item b."""
        pts = torch.from_numpy(local_zyx.astype(np.float32)).to(self.device)[None]
        tg = tok[b:b + 1]
        patch = F.normalize(sample_grid_points(tg, pts, (WIN, WIN, WIN))[0].float(), dim=-1)
        if self.pca is not None:
            patch = patch @ self.pca
        dense = None
        if self.has_dense:
            sk = [s[b:b + 1] for s in skips]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp and self.device.type == "cuda"):
                dense = self.lb.dense_head.forward_points(tg, sk, pts, (WIN, WIN, WIN))["dense"][0].float()
        return patch, dense


def embed_points(sampler: DinovolSampler, read_block: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 zyx: np.ndarray, norm: NormStats | None, batch: int = 4, log=print) -> dict[str, np.ndarray]:
    """Blend-weighted embeddings at every point (N, 3) zyx. read_block(lo, hi) returns raw CT (zyx box)."""
    cores = np.unique(core_cell(zyx), axis=0)
    wins, pts = assign_points(zyx, cores)
    bounds = np.searchsorted(wins, np.arange(len(cores) + 1))
    n = len(zyx)
    pdim = sampler.pca.shape[1] if sampler.pca is not None else int(sampler.lb.backbone.embed_dim)
    acc_p = np.zeros((n, pdim), np.float32)
    acc_d = np.zeros((n, sampler.lb.dense_head.out_dim), np.float32) if sampler.has_dense else None
    wsum = np.zeros(n, np.float32)
    order = np.lexsort((cores[:, 2], cores[:, 1], cores[:, 0]))
    log(f"{n} points, {len(cores)} windows, {len(wins)} point-window pairs")
    for s in range(0, len(order), batch):
        ids = order[s:s + batch]
        starts = window_start(cores[ids])
        blocks = []
        for st in starts:
            raw = read_block(st, st + WIN)
            blocks.append(norm.apply(raw) if norm is not None else normalize_robust(raw))
        tok, skips = sampler.forward(np.stack(blocks).astype(np.float32))
        for b, wid in enumerate(ids):
            sel = pts[bounds[wid]:bounds[wid + 1]]
            if len(sel) == 0:
                continue
            local = zyx[sel] - starts[b]
            w = ramp_weight(local).astype(np.float32)
            patch, dense = sampler.point_features(tok, skips, b, local)
            # sel holds each point once per window, so plain fancy-index accumulation is exact
            acc_p[sel] += patch.cpu().numpy() * w[:, None]
            if dense is not None:
                acc_d[sel] += dense.cpu().numpy() * w[:, None]
            wsum[sel] += w
        if (s // batch) % 200 == 0:
            log(f"  window {s}/{len(order)}")
    out = {"wsum": wsum, "patch": _unit(acc_p)}
    if acc_d is not None:
        out["dense"] = _unit(acc_d)
    return out


def _unit(a: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(a, axis=1, keepdims=True)
    return np.where(nrm > 0, a / np.maximum(nrm, 1e-12), 0.0).astype(np.float32)


def quantise(a: np.ndarray) -> np.ndarray:
    """Unit vectors -> int8 (x127); cosine error ~1e-3."""
    return np.clip(np.round(a * 127.0), -127, 127).astype(np.int8)
