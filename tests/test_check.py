from __future__ import annotations

import json

import numpy as np
import pytest
import tifffile
from PIL import Image

from turncoat import outputs as O
from turncoat.check import CheckError, run_check
from turncoat.cli import main
from turncoat.tifxyz import write_tifxyz

from .conftest import SHAPE, SPACING
from .helpers import write_embeddings
from .synthetic import winding_surface


@pytest.fixture()
def switched(tmp_path):
    s, shift = winding_surface(SHAPE, spacing=SPACING, switch_at=0.5, ramp=1.0)
    seg = write_tifxyz(s, tmp_path / "seg_w001")
    ok = s.valid.copy()
    edge = int(np.nonzero(shift[0] > 0.5)[0].min())
    cols = np.arange(shift.shape[1])
    ramp = SPACING * np.clip((cols - edge + 2) / 3.0, 0.0, 1.0)    # one layer over three columns
    write_embeddings(tmp_path / "emb.npz", ok, np.broadcast_to(ramp, shift.shape))
    return seg, tmp_path / "emb.npz", shift


def quiet(*_):
    pass


def test_check_flags_the_switched_side_and_writes_outputs(switched, m7_zarr, tmp_path):
    seg, emb, shift = switched
    out = tmp_path / "out"
    summ = run_check(seg, embeddings=emb, out=out, prediction=str(m7_zarr), voxel_um=40.0, write_to_segment=True,
                     log=quiet)
    sus = tifffile.imread(out / "suspicion.tif")
    flagged = sus >= 1.0
    side = shift > 0.5
    assert summ["nodes"]["flagged"] > 0 and summ["alarms"]["leave"] >= 1
    smaller, larger = (side, ~side) if side.sum() < (~side).sum() else (~side, side)
    assert flagged[smaller].mean() > 0.9 and flagged[larger].mean() < 0.05
    regs = json.loads((out / "regions.json").read_text())
    assert regs[0]["id"] == 1 and regs[0]["nodes"] > 0 and len(regs[0]["worst_xyz"]) == 3
    pc = json.loads((out / "turncoat_regions.json").read_text())
    assert pc["vc_pointcollections_json_version"] == "1" and len(pc["collections"]["1"]["points"]) == len(regs)
    ink = json.loads((seg / "ink-detections.json").read_text())
    assert {e["local_file"] for e in ink} == {"turncoat_switch_likelihood.png", "turncoat_flagged.png"}
    assert all(e["segment_id"] == "seg_w001" for e in ink)
    png = np.asarray(Image.open(seg / "turncoat_flagged.png"))
    assert png.dtype == np.uint8 and png.shape == shift.shape and set(np.unique(png).tolist()) == {0, O.FLAG_PNG_VALUE}
    assert ((png > 0) == flagged).all()
    assert (seg / "switch_likelihood.tif").exists() and (seg / "x.tif").exists()


def test_alarms_are_cached_and_other_ink_entries_kept(switched, m7_zarr, tmp_path):
    seg, emb, _ = switched
    other = [{"label": "ink", "segment_id": "seg_w001", "local_file": "ink.png"}]
    (seg / "ink-detections.json").write_text(json.dumps(other))
    out = tmp_path / "out"
    a = run_check(seg, embeddings=emb, out=out, prediction=str(m7_zarr), voxel_um=40.0, write_to_segment=True, log=quiet)
    b = run_check(seg, embeddings=emb, out=out, prediction=str(m7_zarr), voxel_um=40.0, write_to_segment=True, log=quiet)
    assert not a["alarms"]["cached"] and b["alarms"]["cached"] and a["alarms"]["points"] == b["alarms"]["points"]
    ink = json.loads((seg / "ink-detections.json").read_text())
    assert [e["local_file"] for e in ink].count("ink.png") == 1 and len(ink) == 3


def test_check_errors_are_clear(switched, tmp_path):
    seg, emb, _ = switched
    with pytest.raises(CheckError, match="turncoat embed"):
        run_check(seg, out=tmp_path / "o", no_prediction=True, log=quiet)
    with pytest.raises(CheckError, match="--volume"):
        run_check(seg, embeddings=emb, out=tmp_path / "o", log=quiet)
    with pytest.raises(CheckError, match="--prediction"):
        run_check(seg, embeddings=emb, volume="/local/scan.zarr", out=tmp_path / "o", log=quiet)


def test_refuses_a_broken_ink_index(switched, m7_zarr, tmp_path):
    seg, emb, _ = switched
    (seg / "ink-detections.json").write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        run_check(seg, embeddings=emb, out=tmp_path / "o", prediction=str(m7_zarr), write_to_segment=True, log=quiet)


def test_cli(switched, m7_zarr, tmp_path, capsys):
    seg, emb, _ = switched
    assert main(["--help"]) == 0 and main(["--version"]) == 0
    assert main(["nope"]) == 2
    assert main(["check", str(seg), "--embeddings", str(emb), "--prediction", str(m7_zarr), "--out",
                 str(tmp_path / "c")]) == 0
    assert (tmp_path / "c" / "summary.json").exists()
    assert main(["check", str(seg), "--out", str(tmp_path / "d"), "--no-prediction"]) == 2
    assert "turncoat embed" in capsys.readouterr().err


def _has_cuda() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


@pytest.mark.skipif(_has_cuda(), reason="a CUDA machine would start a real embedding run")
def test_embed_refuses_without_torch_or_cuda(capsys):
    assert main(["embed", "seg", "--volume", "v", "--ckpt", "c.pt"]) == 2
    assert "embed" in capsys.readouterr().err
