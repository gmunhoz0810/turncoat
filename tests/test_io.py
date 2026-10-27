from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from .synthetic import winding_surface
from turncoat.tifxyz import TifxyzError, read_tifxyz, write_tifxyz
from turncoat.volume import Volume, VolumeError

from .conftest import SHAPE


def test_volume_sample_matches_numpy(spiral, spiral_zarr, rng):
    vol = Volume(spiral_zarr, memory_gb=0.01)
    assert vol.shape == SHAPE
    pts = np.c_[rng.uniform(2, 250, 500), rng.uniform(2, 250, 500), rng.uniform(2, 45, 500)]   # x, y, z
    got = vol.sample(pts)
    want = map_coordinates(spiral.astype(np.float32), [pts[:, 2], pts[:, 1], pts[:, 0]], order=1)
    np.testing.assert_allclose(got, want, atol=1e-3)


def test_volume_read_box_and_outside(spiral, spiral_zarr):
    vol = Volume(spiral_zarr)
    box = vol.read_box((10, 20, 30), (14, 90, 100))
    np.testing.assert_array_equal(box, spiral[10:14, 20:90, 30:100])
    out = vol.read_box((-5, -5, -5), (0, 0, 0))
    assert out.shape == (5, 5, 5) and (out == 0).all()


def test_volume_nan_points_give_nan(spiral_zarr):
    vol = Volume(spiral_zarr)
    v = vol.sample(np.array([[np.nan, 1.0, 1.0], [10.0, 10.0, 10.0]]))
    assert np.isnan(v[0]) and np.isfinite(v[1])


def test_volume_missing_path():
    with pytest.raises(VolumeError):
        Volume("/definitely/not/here.zarr")


def test_tifxyz_roundtrip(tmp_path):
    s, _ = winding_surface(SHAPE)
    valid = s.valid.copy()
    valid[0, :3] = False
    s2 = s.with_xyz(s.xyz, valid)
    p = write_tifxyz(s2, tmp_path / "m.tifxyz")
    r = read_tifxyz(p)
    assert r.shape == s.shape
    assert not r.valid[0, :3].any() and r.valid[1:].all()
    np.testing.assert_allclose(r.xyz[r.valid], s2.xyz[s2.valid], atol=1e-3)
    assert abs(r.grid_step() - 4.0) < 0.6
    with pytest.raises(TifxyzError):
        write_tifxyz(r, p)        # refuses to overwrite its own source


def test_tifxyz_rejects_garbage(tmp_path):
    (tmp_path / "bad").mkdir()
    with pytest.raises(TifxyzError):
        read_tifxyz(tmp_path / "bad")
