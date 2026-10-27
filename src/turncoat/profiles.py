"""CT intensity profiles along each vertex normal: the raw evidence every Turncoat test works from."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .geometry import normals
from .tifxyz import Surface
from .volume import Volume


@dataclass(frozen=True)
class Profiles:
    """prof[i, k] = CT at node rc[i] + t[k] * normal[i]; t runs -K..K voxels in steps of 1."""

    rc: np.ndarray        # (n, 2) grid row/col of each profiled node
    xyz: np.ndarray       # (n, 3) node positions
    nrm: np.ndarray       # (n, 3) unit normals
    prof: np.ndarray      # (n, 2K+1) float32
    K: int
    shape: tuple[int, int]

    @property
    def t(self) -> np.ndarray:
        return np.arange(-self.K, self.K + 1, dtype=np.float32)

    def index_grid(self) -> np.ndarray:
        """(H, W) array of profile row indices, -1 where a node has no profile."""
        idx = -np.ones(self.shape, np.int64)
        idx[self.rc[:, 0], self.rc[:, 1]] = np.arange(len(self.rc))
        return idx

    def shifted(self, d: np.ndarray) -> Profiles:
        """Profiles of the same nodes moved by d[i] voxels along their normals (sub-voxel, linear).

        This is how the self-test plants a drift without fetching new CT: moving a node along its own normal
        by d samples the same line, offset by d. The window shrinks by ceil(max |d|) on both sides.
        """
        dmax = int(np.ceil(np.nanmax(np.abs(d)))) if len(d) else 0
        if dmax >= self.K - 4:
            raise ValueError(f"shift {dmax} vx too large for profile half-width {self.K}")
        K2 = self.K - dmax
        t2 = np.arange(-K2, K2 + 1, dtype=np.float64)
        pos = t2[None, :] + d[:, None] + self.K
        i0 = np.clip(np.floor(pos).astype(int), 0, 2 * self.K - 1)
        f = (pos - i0).astype(np.float32)
        rows = np.arange(len(d))[:, None]
        p = self.prof[rows, i0] * (1 - f) + self.prof[rows, i0 + 1] * f
        return Profiles(rc=self.rc, xyz=self.xyz + d[:, None] * self.nrm, nrm=self.nrm, prof=p.astype(np.float32),
                        K=K2, shape=self.shape)


def sample_profiles(surface: Surface, volume: Volume, K: int = 40, smooth: float = 1.0,
                    progress: bool = False) -> Profiles:
    """Sample CT along every valid node's normal, t = -K..K voxels (trilinear), lightly smoothed along t."""
    return sample_profiles_with(surface, volume, K, smooth, progress)[0]


def sample_profiles_with(surface: Surface, volume: Volume, K: int = 40, smooth: float = 1.0, progress: bool = False,
                         extra: list[np.ndarray] | None = None) -> tuple[Profiles, list[np.ndarray]]:
    """Like sample_profiles, plus the CT at any `extra` point arrays (..., 3), read in the same chunk pass."""
    nrm, ok = normals(surface.xyz, surface.valid)
    rc = np.argwhere(ok)
    if len(rc) == 0:
        raise ValueError("surface has no vertex with a defined normal")
    p = surface.xyz[ok]
    n = nrm[ok]
    t = np.arange(-K, K + 1, dtype=np.float64)
    pts = p[:, None, :] + t[None, :, None] * n[:, None, :]
    extra = extra or []
    parts = [pts.reshape(-1, 3)] + [e.reshape(-1, 3) for e in extra]
    vals = volume.sample(np.concatenate(parts), order=1, progress=progress).astype(np.float32)
    sizes = np.cumsum([len(x) for x in parts])[:-1]
    chunks = np.split(vals, sizes)
    prof = chunks[0].reshape(pts.shape[:2])
    extra_vals = [c.reshape(e.shape[:-1]) for c, e in zip(chunks[1:], extra)]
    prof = np.nan_to_num(prof, nan=0.0)
    if smooth > 0:
        prof = gaussian_filter1d(prof, smooth, axis=1)
    return Profiles(rc=rc, xyz=p, nrm=n, prof=prof, K=K, shape=surface.shape), extra_vals
