"""The benchmark: the published flags reproduce the README's numbers, and the scorer follows the stated rules."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from turncoat.bench import BenchError, load_bench, read_flags, score, write_vc3d
from turncoat.cli import main
from turncoat.tifxyz import Surface, write_tifxyz

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "bench"
SCROLLS = {"PHerc0139", "PHerc0500P2", "PHerc0814", "PHerc0841"}


@pytest.fixture(scope="module")
def bench():
    return load_bench(BENCH)


def _f1(res, group, part):
    return res["pooled"][group][part]["F1"]


def test_manifest_meshes_and_size(bench):
    m = bench.manifest
    assert len(m) == 14 and sum(v["split"] == "test" for v in m.values()) == 12
    assert {v["scroll"] for v in m.values()} == SCROLLS
    assert all(v["tifxyz_url"].startswith("https://vesuvius-challenge-open-data.s3") for v in m.values())
    size = sum(p.stat().st_size for p in BENCH.rglob("*") if p.is_file())
    assert size < 30 * 2 ** 20


def test_label_counts_match_manifest(bench):
    for k, v in bench.manifest.items():
        g = bench.labels[k]
        assert (int((g == 0).sum()), int((g == 1).sum())) == (v["off"], v["on"]), k
        assert len(bench.pairs[k][1]) == v["fibre_pairs"] and int(bench.pairs[k][1].sum()) == v["fibre_pairs_different"]
        assert len(bench.verified[k]) == v["verified_nodes"]
    assert sum(len(v[1]) for v in bench.human.values()) == 230


def test_turncoat_flags_reproduce_published_numbers(bench):
    res = score(BENCH / "flags/turncoat", bench)
    assert (_f1(res, "test", "node_exact"), _f1(res, "test", "fibre_pairs"), _f1(res, "test", "verified_nodes")) == \
        (0.537, 0.234, 0.416)
    assert [_f1(res, s, "node_exact") for s in sorted(SCROLLS)] == [0.499, 0.461, 0.470, 0.768]
    stored = json.loads((ROOT / "evidence/fresh12_node_exact.json").read_text())["per_mesh"]
    for k, row in res["per_mesh"].items():
        assert row["node_exact"] == stored[k]["turncoat"], k


def test_doctor_flags_reproduce_published_numbers(bench):
    res = score(BENCH / "flags/tifxyz-doctor", bench)
    assert (_f1(res, "test", "node_exact"), _f1(res, "test", "fibre_pairs"), _f1(res, "test", "verified_nodes")) == \
        (0.032, 0.041, 0.020)
    stored = json.loads((ROOT / "evidence/fresh12_node_exact.json").read_text())["per_mesh"]
    for k, row in res["per_mesh"].items():
        assert row["node_exact"] == stored[k]["tifxyz-doctor"], k


def test_dev_fit_reproduces(bench):
    res = score(BENCH / "flags/turncoat", bench, split="dev")
    ne = res["pooled"]["dev"]["node_exact"]
    assert (ne["R"], ne["P"], ne["F1"]) == (88.8, 92.5, 0.906)        # evidence/dev_fit.json, first row
    assert "PHerc0139" not in res["pooled"]                           # dev meshes never pool with the test scrolls


# --- the rules on a tiny synthetic benchmark ---------------------------------------------------------------------------

def _tiny(tmp_path: Path) -> Path:
    b = tmp_path / "bench"
    (b / "labels").mkdir(parents=True)
    lab = np.array([[1, 1, 0, 0], [1, 255, 0, 1]], np.uint8)
    Image.fromarray(lab).save(b / "labels/m1.png")
    (b / "manifest.json").write_text(json.dumps({"m1": {"scroll": "S", "split": "test", "grid_shape": [2, 4],
                                                         "x_tif_sha256": "0" * 64}}))
    (b / "fibre_pairs.csv").write_text("mesh,row_a,col_a,row_b,col_b,relation\n"
                                       "m1,0,0,0,2,different\nm1,0,0,0,1,same\nm1,0,2,0,3,same\nm1,1,0,1,2,different\n")
    (b / "verified_nodes.csv").write_text("mesh,row,col,label\nm1,0,2,0\nm1,0,1,1\n")
    (b / "human_fibre_nodes.csv").write_text("mesh,row,col,label\nm1,0,0,0\nm1,0,2,0\n")
    return b


def test_rules_on_tiny_bench(tmp_path):
    b = load_bench(_tiny(tmp_path))
    flags = tmp_path / "flags"
    flags.mkdir()
    np.save(flags / "m1.npy", np.array([[0, 1, 1, 0], [0, 1, 0, 0]], bool))
    r = score(flags, b)["per_mesh"]["m1"]
    assert r["node_exact"] == {"tp": 1, "fp": 1, "off": 3, "on": 4}          # (1,1) unlabelled: not scored
    assert r["fibre_pairs"] == {"tp": 1, "fp": 2, "pos": 2}                  # exactly one node flagged = two sheets
    assert r["verified_nodes"] == {"tp": 1, "fp": 1, "pos": 1}
    assert r["human_fibre_nodes"] == {"tp": 1, "fp": 0, "pos": 2}


@pytest.mark.parametrize("fmt", ["png", "npy", "npz", "tif", "folder"])
def test_flag_formats(tmp_path, fmt):
    b = load_bench(_tiny(tmp_path))
    f = np.array([[0, 0, 1, 1], [0, 0, 1, 0]], bool)
    d = tmp_path / "flags"
    d.mkdir()
    if fmt == "png":
        Image.fromarray((f * 164).astype(np.uint8)).save(d / "m1.png")
    elif fmt == "npy":
        np.save(d / "m1.npy", f)
    elif fmt == "npz":
        np.savez_compressed(d / "m1.npz", flags=f)
    elif fmt == "tif":
        tifffile.imwrite(d / "m1.tif", np.where(f, 1.0, np.nan).astype(np.float32))
    else:
        (d / "m1").mkdir()
        tifffile.imwrite(d / "m1/suspicion.tif", np.where(f, 1.0, 0.0).astype(np.float32))
    assert score(d, b)["per_mesh"]["m1"]["node_exact"] == {"tp": 3, "fp": 0, "off": 3, "on": 4}


def test_bad_inputs(tmp_path):
    b = load_bench(_tiny(tmp_path))
    d = tmp_path / "flags"
    d.mkdir()
    with pytest.raises(BenchError, match="no flags"):
        score(d, b)
    np.save(d / "m1.npy", np.zeros((3, 4), bool))
    with pytest.raises(BenchError, match="mesh grid"):
        score(d, b)
    with pytest.raises(BenchError, match="unknown mesh"):
        score(d, b, meshes=["nope"])
    with pytest.raises(BenchError, match="not a benchmark"):
        load_bench(tmp_path)
    with pytest.raises(BenchError):
        read_flags(d / "m1.txt", (2, 4))


def test_cli_score_and_errors(tmp_path, capsys):
    b = _tiny(tmp_path)
    d = tmp_path / "flags"
    d.mkdir()
    np.save(d / "m1.npy", np.ones((2, 4), bool))
    out = tmp_path / "res.json"
    assert main(["bench", "score", "--flags", str(d), "--bench", str(b), "--json", str(out)]) == 0
    assert json.loads(out.read_text())["pooled"]["test"]["node_exact"]["R"] == 100.0
    assert main(["bench", "score", "--flags", str(tmp_path / "none"), "--bench", str(b)]) == 2
    assert "not a folder" in capsys.readouterr().err


def test_write_vc3d(tmp_path):
    b = _tiny(tmp_path)
    xyz = np.stack([*np.meshgrid(np.arange(4.0), np.arange(2.0), indexing="xy"), np.zeros((2, 4))], -1) + 1.0
    seg = write_tifxyz(Surface(xyz, np.ones((2, 4), bool), {"uuid": "m1"}), tmp_path / "m1.tifxyz")
    mf = json.loads((b / "manifest.json").read_text())
    mf["m1"]["x_tif_sha256"] = hashlib.sha256((seg / "x.tif").read_bytes()).hexdigest()
    (b / "manifest.json").write_text(json.dumps(mf))
    (seg / "ink-detections.json").write_text(json.dumps([{"label": "other", "local_file": "other.png"}]))
    written = write_vc3d(seg, "m1", load_bench(b))
    assert written == ["bench_truth_off.png", "bench_truth_unlabelled.png", "ink-detections.json"]
    off = np.asarray(Image.open(seg / "bench_truth_off.png"))
    assert (off > 0).tolist() == [[False, False, True, True], [False, False, True, False]]
    ink = json.loads((seg / "ink-detections.json").read_text())
    assert [e["local_file"] for e in ink] == ["other.png", "bench_truth_off.png", "bench_truth_unlabelled.png"]
    with pytest.raises(BenchError, match="unknown mesh"):
        write_vc3d(seg, "m2", load_bench(b))
