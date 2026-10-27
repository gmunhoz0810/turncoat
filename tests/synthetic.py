"""Synthetic rolled-papyrus volumes and traced surfaces with planted sheet switches (for tests and demos).

The volume is an Archimedean spiral of bright sheets (layer spacing `spacing` voxels) extruded along z, with
per-winding brightness, slow thickness variation and noise, so neighbouring layers are similar but not identical
(as in real scans). A surface follows one winding; `switch_at` makes it slide onto the next winding over `ramp`
grid columns: a sheet switch whose true location is known.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from turncoat.tifxyz import Surface


def spiral_volume(shape=(64, 256, 256), spacing: float = 12.0, r0: float = 20.0, sigma: float = 2.0,
                  noise: float = 12.0, seed: int = 0) -> np.ndarray:
    """uint8 (z, y, x) volume of a spiral of sheets centred in the xy plane."""
    rng = np.random.default_rng(seed)
    Z, Y, X = shape
    yy, xx = np.mgrid[0:Y, 0:X].astype(np.float64)
    cy, cx = (Y - 1) / 2, (X - 1) / 2
    r = np.hypot(yy - cy, xx - cx)
    th = np.mod(np.arctan2(yy - cy, xx - cx), 2 * np.pi)
    turns = (r - r0) / spacing - th / (2 * np.pi)          # winding coordinate
    k = np.round(turns)
    d = (turns - k) * spacing                               # radial distance to the nearest sheet
    nk = int(np.nanmax(k)) + 3
    bright = rng.uniform(0.55, 1.0, nk + 3)
    vol = np.empty(shape, np.uint8)
    for z in range(Z):
        wobble = 1.0 + 0.25 * np.sin(th * 3 + z / 9.0 + k)
        sig = sigma * wobble
        b = bright[np.clip(k.astype(int), 0, nk + 2)]
        img = 30 + 190 * b * np.exp(-0.5 * (d / sig) ** 2) * (k >= 0)
        img = img + rng.normal(0, noise, img.shape)
        vol[z] = np.clip(img, 0, 255).astype(np.uint8)
    return vol


def winding_surface(shape=(64, 256, 256), spacing: float = 12.0, r0: float = 20.0, winding: int = 4,
                    step: float = 4.0, theta_span=(0.3, 5.6), z_margin: int = 4, switch_at: float | None = None,
                    ramp: float = 3.0, switch_layers: int = 1) -> tuple[Surface, np.ndarray]:
    """Surface on one winding (rows = z, cols = arclength). Returns (surface, planted layer shift per node)."""
    Z, Y, X = shape
    cy, cx = (Y - 1) / 2, (X - 1) / 2
    zs = np.arange(z_margin, Z - z_margin, step)
    r_mid = r0 + spacing * (winding + np.mean(theta_span) / (2 * np.pi))
    n_cols = int((theta_span[1] - theta_span[0]) * r_mid / step)
    th = np.linspace(theta_span[0], theta_span[1], n_cols)
    cols = np.arange(n_cols, dtype=float)
    shift = np.zeros(n_cols)
    if switch_at is not None:
        shift = switch_layers * np.clip((cols - switch_at * n_cols) / max(ramp, 1e-6), 0, 1)
    r = r0 + spacing * (winding + shift + th / (2 * np.pi))
    x = cx + r * np.cos(th)
    y = cy + r * np.sin(th)
    H, W = len(zs), n_cols
    xyz = np.stack([np.broadcast_to(x, (H, W)), np.broadcast_to(y, (H, W)), np.broadcast_to(zs[:, None], (H, W))], -1)
    valid = np.ones((H, W), bool)
    meta = {"format": "tifxyz", "scale": [1.0 / step, 1.0 / step], "type": "seg", "uuid": "synthetic"}
    return Surface(xyz=xyz.astype(np.float64).copy(), valid=valid, meta=meta), np.broadcast_to(shift, (H, W)).copy()


def write_zarr(vol: np.ndarray, path: str | Path, chunks=(32, 32, 32)) -> Path:
    """Write a minimal uncompressed Zarr v2 OME-Zarr group (level 0 only) to a local directory."""
    root = Path(path)
    arr = root / "0"
    arr.mkdir(parents=True, exist_ok=True)
    (root / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (root / ".zattrs").write_text(json.dumps({"multiscales": [{"version": "0.4", "datasets": [{"path": "0"}],
                                                               "axes": [{"name": a, "type": "space"} for a in "zyx"]}]}))
    meta = {"zarr_format": 2, "shape": list(vol.shape), "chunks": list(chunks), "dtype": "|u1", "fill_value": 0,
            "order": "C", "filters": None, "compressor": None, "dimension_separator": "/"}
    (arr / ".zarray").write_text(json.dumps(meta))
    for iz in range(0, vol.shape[0], chunks[0]):
        for iy in range(0, vol.shape[1], chunks[1]):
            for ix in range(0, vol.shape[2], chunks[2]):
                block = np.zeros(chunks, np.uint8)
                sub = vol[iz:iz + chunks[0], iy:iy + chunks[1], ix:ix + chunks[2]]
                block[:sub.shape[0], :sub.shape[1], :sub.shape[2]] = sub
                p = arr / str(iz // chunks[0]) / str(iy // chunks[1])
                p.mkdir(parents=True, exist_ok=True)
                (p / str(ix // chunks[2])).write_bytes(block.tobytes())
    return root
