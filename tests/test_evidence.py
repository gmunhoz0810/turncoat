"""The README's numbers are recomputed from the stored counts and must equal the stored summaries."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "evidence"


def _score():
    spec = importlib.util.spec_from_file_location("evidence_score", ROOT / "score.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_node_exact_summaries_recompute():
    stored = json.loads((ROOT / "fresh12_node_exact.json").read_text())["summary"]
    got = _score().node_exact()
    for group, row in stored.items():
        for det, v in row.items():
            assert {k: got[group][det][k] for k in ("R", "P", "F1")} == v, (group, det)


def test_fibre_numbers_recompute():
    stored = json.loads((ROOT / "fibre_check.json").read_text())["fresh12"]
    got = _score().fibres()
    for part in ("fibre_pairs", "verified_nodes"):
        for det, v in stored[part].items():
            assert {k: got[part][det][k] for k in ("R", "P", "F1")} == {k: v[k] for k in ("R", "P", "F1")}


def test_headline_numbers():
    ne, fb = _score().node_exact(), _score().fibres()
    assert ne["fresh12"]["turncoat"]["F1"] == 0.537 and ne["fresh12"]["tifxyz-doctor"]["F1"] == 0.032
    assert fb["fibre_pairs"]["turncoat"]["F1"] == 0.234 and fb["fibre_pairs"]["tifxyz-doctor"]["F1"] == 0.041
    assert fb["verified_nodes"]["turncoat"]["F1"] == 0.416 and fb["verified_nodes"]["tifxyz-doctor"]["F1"] == 0.020
