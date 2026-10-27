"""Read and write tifxyz quad meshes (the VC3D surface format).

A tifxyz is a directory holding x.tif, y.tif and z.tif (float32, one vertex per grid cell, in voxel
coordinates of the volume the mesh was traced on) plus meta.json. A vertex is missing when its
coordinates are -1 (or NaN, or all zero), or when an optional mask.tif marks it 0.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile


class TifxyzError(ValueError):
    """The directory is not a readable tifxyz mesh."""


@dataclass(frozen=True)
class Surface:
    """A quad-mesh surface: xyz (H, W, 3) float64 in voxels (NaN where missing) and a validity mask."""

    xyz: np.ndarray
    valid: np.ndarray
    meta: dict = field(default_factory=dict)
    path: Path | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.valid.shape

    @property
    def render_scale(self) -> float:
        """Render pixels per grid node (1 / meta scale); 20 for a VC3D mesh saved at scale 0.05."""
        scale = self.meta.get("scale")
        if isinstance(scale, (list, tuple)) and scale and float(scale[0]) > 0:
            return 1.0 / float(scale[0])
        return 1.0

    def grid_step(self) -> float:
        """Median 3-D distance between grid-adjacent vertices, in voxels."""
        p = self.xyz
        steps = []
        for axis in (0, 1):
            a = np.take(p, range(p.shape[axis] - 1), axis=axis)
            b = np.take(p, range(1, p.shape[axis]), axis=axis)
            d = np.linalg.norm(b - a, axis=-1)
            steps.append(d[np.isfinite(d)])
        allsteps = np.concatenate(steps)
        if allsteps.size == 0:
            raise TifxyzError("mesh has no pair of adjacent valid vertices")
        return float(np.median(allsteps))

    def with_xyz(self, xyz: np.ndarray, valid: np.ndarray | None = None, **meta_updates) -> Surface:
        """A new Surface with replaced coordinates (never mutates this one)."""
        v = self.valid if valid is None else valid
        new_xyz = np.where(v[..., None], xyz, np.nan)
        return Surface(xyz=new_xyz, valid=v.copy(), meta={**self.meta, **meta_updates}, path=None)


def read_tifxyz(path: str | Path) -> Surface:
    p = Path(path)
    if not p.is_dir():
        raise TifxyzError(f"{p} is not a directory")
    missing = [c for c in "xyz" if not (p / f"{c}.tif").exists()]
    if missing:
        raise TifxyzError(f"{p} lacks {', '.join(c + '.tif' for c in missing)}")
    try:
        chans = [tifffile.imread(p / f"{c}.tif").astype(np.float64) for c in "xyz"]
    except Exception as exc:  # tifffile raises several types; report them uniformly
        raise TifxyzError(f"cannot read {p}: {exc}") from exc
    if not (chans[0].shape == chans[1].shape == chans[2].shape) or chans[0].ndim != 2:
        raise TifxyzError(f"{p}: x/y/z.tif must be 2-D and of equal shape")
    xyz = np.stack(chans, -1)
    valid = np.isfinite(xyz).all(-1)
    valid &= ~np.all(xyz == -1, axis=-1)
    valid &= ~np.all(xyz == 0, axis=-1)
    valid &= np.all(xyz >= 0, axis=-1)
    if (p / "mask.tif").exists():
        mask = tifffile.imread(p / "mask.tif")
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.shape == valid.shape:
            valid &= mask > 0
    meta = {}
    if (p / "meta.json").exists():
        try:
            meta = json.loads((p / "meta.json").read_text())
        except json.JSONDecodeError as exc:
            raise TifxyzError(f"{p}/meta.json is not valid JSON: {exc}") from exc
    if not valid.any():
        raise TifxyzError(f"{p} has no valid vertices")
    return Surface(xyz=np.where(valid[..., None], xyz, np.nan), valid=valid, meta=meta, path=p)


def write_tifxyz(surface: Surface, path: str | Path, overwrite: bool = False) -> Path:
    """Write a Surface as a tifxyz directory; missing vertices become -1. Refuses to overwrite its own source."""
    dst = Path(path)
    if surface.path is not None and dst.resolve() == Path(surface.path).resolve():
        raise TifxyzError("refusing to overwrite the source mesh")
    if dst.exists():
        if not overwrite:
            raise TifxyzError(f"{dst} exists (pass overwrite=True)")
        if not (dst / "x.tif").exists():
            raise TifxyzError(f"{dst} exists and does not look like a tifxyz; not touching it")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for k, c in enumerate("xyz"):
        arr = np.where(surface.valid, surface.xyz[..., k], -1.0).astype(np.float32)
        tifffile.imwrite(dst / f"{c}.tif", arr)
    meta = {**surface.meta, "format": "tifxyz"}
    meta.setdefault("scale", [0.05, 0.05])
    (dst / "meta.json").write_text(json.dumps(meta, indent=1))
    return dst
