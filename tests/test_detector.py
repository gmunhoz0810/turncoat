from __future__ import annotations

import numpy as np
import pytest

from turncoat import detector as DT
from turncoat.embeddings import EmbeddingError, dinovol_offset, load_embeddings, sheet_pitch

from .helpers import write_embeddings


def grid(H=60, W=80, step=20.0):
    rr, cc = np.mgrid[0:H, 0:W].astype(float)
    xyz = np.stack([cc * step, rr * step, np.full_like(rr, 500.0)], -1)
    return xyz, np.ones((H, W), bool)


def test_min_nodes_from_grid_spacing():
    xyz, _ = grid(step=20.0)
    node_mm2 = (20 * 9.362e-3) ** 2
    assert DT.min_nodes(xyz, 9.362, 1.0) == int(np.ceil(1.0 / node_mm2))


def test_flags_need_a2_area_and_an_alarm():
    xyz, valid = grid()
    off = np.zeros(valid.shape)
    off[10:30, 10:30] = 12.0                     # big region, 0.6 layers off at pitch 20
    off[50:52, 70:72] = 12.0                     # 4 nodes: under 1 mm2
    ok = valid.copy()
    alarm = xyz[20, 20][None]                    # an alarm inside the big region
    f = DT.flags(off, 20.0, ok, xyz, valid, 9.362, alarm)
    assert f[10:30, 10:30].all() and not f[50:52, 70:72].any() and f.sum() == 400
    assert not DT.flags(off, 20.0, ok, xyz, valid, 9.362, np.zeros((0, 3))).any()      # no alarm support: dropped
    far = xyz[0, 79][None] + [0, 0, 500.0]
    assert not DT.flags(off, 20.0, ok, xyz, valid, 9.362, far).any()
    small = np.where(off > 0, 5.0, 0.0)                                                  # 0.25 layers: below A2_MIN
    assert not DT.flags(small, 20.0, ok, xyz, valid, 9.362, alarm).any()


def test_flags_validate_shapes_and_pitch():
    xyz, valid = grid()
    with pytest.raises(ValueError):
        DT.flags(np.zeros((3, 3)), 20.0, valid, xyz, valid, 9.362, np.zeros((0, 3)))
    with pytest.raises(ValueError):
        DT.flags(np.zeros(valid.shape), 0.0, valid, xyz, valid, 9.362, np.zeros((0, 3)))


def test_likelihood_scale_and_nan():
    valid = np.ones((2, 3), bool)
    ok = valid.copy()
    ok[0, 0] = False
    off = np.array([[0.0, 5.0, 10.0], [20.0, -10.0, 0.0]])
    lk = DT.likelihood(off, 20.0, ok, valid)
    assert np.isnan(lk[0, 0]) and lk[0, 1] == pytest.approx(0.5) and lk[0, 2] == pytest.approx(1.0)
    assert lk[1, 0] == 1.0 and lk[1, 1] == pytest.approx(1.0) and lk.dtype == np.float32


def test_embeddings_offset_finds_the_shifted_sheet(tmp_path):
    H, W = 24, 30
    ok = np.ones((H, W), bool)
    shift = np.zeros((H, W))
    shift[:, 17] = 4.0                           # a 10 vx step over two columns (the tracker caps one edge at 8 vx)
    shift[:, 18:] = 10.0
    f = write_embeddings(tmp_path / "emb.npz", ok, shift)
    off, ok2, pitch = dinovol_offset(f, (H, W))
    assert ok2.all() and pitch > 0
    left, right = np.nanmedian(off[:, :15]), np.nanmedian(off[:, 20:])
    assert abs(abs(right - left) - 10.0) < 2.0
    assert sheet_pitch(load_embeddings(f, (H, W))) == pytest.approx(pitch)


def test_embeddings_reject_a_wrong_grid(tmp_path):
    ok = np.ones((5, 6), bool)
    f = write_embeddings(tmp_path / "emb.npz", ok, np.zeros((5, 6)))
    with pytest.raises(EmbeddingError):
        dinovol_offset(f, (6, 6))
    np.savez(tmp_path / "bad.npz", ok=ok)
    with pytest.raises(EmbeddingError):
        dinovol_offset(tmp_path / "bad.npz", (5, 6))
