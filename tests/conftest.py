from __future__ import annotations

import numpy as np
import pytest

from .synthetic import spiral_volume, write_zarr

SHAPE = (48, 256, 256)
SPACING = 12.0


@pytest.fixture(scope="session")
def spiral():
    return spiral_volume(SHAPE, spacing=SPACING, seed=1)


@pytest.fixture(scope="session")
def spiral_zarr(spiral, tmp_path_factory):
    root = tmp_path_factory.mktemp("vol") / "spiral.zarr"
    write_zarr(spiral, root)
    return root


@pytest.fixture(scope="session")
def m7_zarr(tmp_path_factory):
    clean = spiral_volume(SHAPE, spacing=SPACING, seed=1, noise=0.0)
    return write_zarr(np.where(clean > 120, 255, 0).astype(np.uint8), tmp_path_factory.mktemp("m7") / "m7.zarr")


@pytest.fixture()
def rng():
    return np.random.default_rng(0)
