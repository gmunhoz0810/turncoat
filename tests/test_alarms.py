from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import binary_erosion

from turncoat.alarms.channel import ChannelOutput, midrank_cdf, reference_sample
from turncoat.alarms.check import leave_step_alarms
from turncoat.alarms.fusion3 import Calibration3, Calibration3Error, cluster_centroids, component_alarms, nodes_for_area
from turncoat.alarms.instances import erode6
from turncoat.alarms.leave import LeaveScan, offsets_from_records, scan, tiles
from turncoat.alarms.predictions import find_prediction, parse_volume
from turncoat.alarms.step import StepParams, spans_in_nodes, step_channel, step_fields
from turncoat.tifxyz import Surface
from turncoat.volume import Volume

from .conftest import SHAPE, SPACING
from .synthetic import winding_surface

B = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"


def plane(H=40, W=60, step=20.0) -> Surface:
    rr, cc = np.mgrid[0:H, 0:W].astype(float)
    xyz = np.stack([cc * step, rr * step, np.full_like(rr, 500.0)], -1)
    return Surface(xyz=xyz, valid=np.ones((H, W), bool), meta={"scale": [0.05, 0.05]})


def test_midrank_cdf_ties_sit_in_the_middle():
    ref = np.sort(np.r_[np.zeros(90), np.linspace(1, 10, 10)])
    u = midrank_cdf(ref, np.array([0.0, 10.0, 100.0, np.nan]))
    assert u[0] == pytest.approx(0.45)
    assert u[1] == pytest.approx(0.995)
    assert u[2] == 1.0 and np.isnan(u[3])
    with pytest.raises(ValueError):
        midrank_cdf(np.array([]), u)


def test_reference_sample_keeps_tie_share():
    v = np.r_[np.zeros(80000), np.ones(20000), np.nan]
    ref = reference_sample(v, size=1001)
    assert len(ref) == 1001 and np.all(np.diff(ref) >= 0)
    assert np.mean(ref == 0) == pytest.approx(0.8, abs=0.01)


def test_step_flags_an_abrupt_step_not_a_plane():
    s = plane()
    stepped = s.xyz.copy()
    stepped[:, 30:, 2] += 14.0
    out = step_channel(s.with_xyz(stepped), StepParams(scales_vx=(1.0, 1.0, 1.0), spans_vx=(20.0, 40.0, 80.0)))
    assert np.nanmax(out.score[:, 28:32]) >= 13.0
    assert np.nanmax(out.score[:, :25]) < 1e-6
    assert spans_in_nodes(plane(step=20.0), (20.0, 40.0, 80.0)) == (1, 2, 4)
    assert step_fields(plane(step=50.0), (20.0, 40.0, 80.0)).shape[0] == 3


def test_step_requires_calibrated_scales():
    with pytest.raises(ValueError):
        step_channel(plane(), StepParams(scales_vx=None))


def test_channel_output_validates_shapes():
    with pytest.raises(ValueError):
        ChannelOutput("x", np.zeros(3))


def test_erode6_matches_scipy():
    b = np.random.default_rng(3).random((20, 23, 17)) > 0.25
    assert np.array_equal(erode6(b), binary_erosion(b))


def test_tiles_cover_the_grid_with_overlap():
    covered = np.zeros((25, 31), bool)
    for r, c in tiles(25, 31):
        covered[r:r + 10, c:c + 10] = True
    assert covered.all()


def test_leave_finds_a_switch_onto_a_parallel_instance(m7_zarr):
    vol = Volume(str(m7_zarr), memory_gb=0.2)
    clean, _ = winding_surface(SHAPE, spacing=SPACING)
    sw, shift = winding_surface(SHAPE, spacing=SPACING, switch_at=0.5, ramp=1.0)
    assert len(scan(clean, vol).alarms(1, 50.0)) == 0
    r1 = scan(sw, vol)
    al = r1.alarms(3, 50.0)
    assert len(al) >= 3
    edge = np.nonzero(shift[0] > 0.5)[0].min()
    assert np.all(np.abs(al[:, 6] - edge) <= 4)
    assert np.nanmedian(np.abs(offsets_from_records(r1.records, sw.shape))) == pytest.approx(SPACING, abs=3.0)


def test_leave_alarm_support_counts_neighbours():
    rec = np.zeros((4, 10))
    rec[:, :3] = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [100, 0, 0]]
    rec[:, 8] = 1
    s = LeaveScan(rec, np.zeros((2, 2)), np.zeros((2, 2)), 0)
    assert len(s.alarms(3, 5.0)) == 3 and len(s.alarms(1, 5.0)) == 4


def test_component_and_cluster_alarms():
    xyz = np.stack(np.meshgrid(np.arange(10.0), np.arange(10.0), indexing="ij") + (np.zeros((10, 10)),), -1)
    m = np.zeros((10, 10), bool)
    m[2, 2:5] = True
    a, keep = component_alarms(m.astype(float), m, xyz, 1, "step")
    assert len(a) >= 1 and keep.sum() == 3
    assert component_alarms(m.astype(float), m, xyz, 8, "step")[0] == []
    c, _ = cluster_centroids(m, xyz, 3, "offpred")
    assert len(c) == 1 and c[0].rc == (2, 3)
    assert nodes_for_area(xyz * 20, 2.0, 0.0096) == int(np.ceil(2.0 / (20 * 0.0096) ** 2))


def test_calibration_loads():
    cal = Calibration3.load()
    p = cal.points[cal.primary]
    assert p.a_support >= 1 and 0.9 < p.step_u < 1
    with pytest.raises(Calibration3Error):
        Calibration3.load("/nonexistent.json")


def test_leave_step_alarms_on_a_switched_trace(m7_zarr):
    vol = Volume(str(m7_zarr), memory_gb=0.2)
    clean, _ = winding_surface(SHAPE, spacing=SPACING)
    sw, shift = winding_surface(SHAPE, spacing=SPACING, switch_at=0.5, ramp=1.0)
    a0, i0 = leave_step_alarms(clean, 9.362, vol, log=lambda *_: None)
    a1, i1 = leave_step_alarms(sw, 9.362, vol, log=lambda *_: None)
    assert i0["leave"] == 0 and i1["leave"] >= 1
    assert a1.shape[1] == 3 and len(a1) == i1["leave"] + i1["step"]
    a2, i2 = leave_step_alarms(sw, 9.362, None, log=lambda *_: None)
    assert i2["leave"] == 0 and len(a2) == i2["step"]


def test_prediction_lookup():
    u, _ = find_prediction(f"{B}/PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr")
    assert u.endswith("20250728140407-surface-20260413222639-surface-m7-L0-th0.2.zarr")
    u, why = find_prediction("/local/volume.zarr")
    assert u is None and "--prediction" in why
    assert parse_volume(f"{B}/PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr/1") == \
        ("PHerc0139", "20260102150214", 1)
