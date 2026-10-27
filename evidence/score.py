"""Recompute every number in the README's Evidence section from the stored counts.

  python evidence/score.py            prints the tables
Node exact (fresh12_node_exact.json): pooled over meshes, R = sum tp / sum off, P = sum tp / sum (tp + fp).
Fibre check (fibre_check.json): R = tp / pos, P = tp / (tp + fp)."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DETECTORS = ("turncoat", "tifxyz-doctor")


def prf(tp: int, fp: int, pos: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / pos if pos else 0.0
    return {"R": round(100 * r, 1), "P": round(100 * p, 1), "F1": round(2 * p * r / (p + r), 3) if tp else 0.0}


def node_exact() -> dict:
    d = json.loads((HERE / "fresh12_node_exact.json").read_text())
    groups: dict = {"fresh12": list(d["per_mesh"])}
    for k, v in d["per_mesh"].items():
        groups.setdefault(v["scroll"], []).append(k)
    out = {}
    for g, keys in groups.items():
        out[g] = {}
        for det in DETECTORS:
            tp, fp, off = (sum(d["per_mesh"][k][det][f] for k in keys) for f in ("tp", "fp", "off"))
            out[g][det] = {**prf(tp, fp, off), "meshes": len(keys), "scored_nodes": sum(d["per_mesh"][k][det]["off"] +
                                                                                          d["per_mesh"][k][det]["on"] for k in keys)}
    return out


def fibres() -> dict:
    d = json.loads((HERE / "fibre_check.json").read_text())["fresh12"]
    return {part: {det: {**prf(v["tp"], v["fp"], v["pos"]), "n": v["pos"]} for det, v in d[part].items()}
            for part in ("fibre_pairs", "verified_nodes")}


def main() -> int:
    ne = node_exact()
    print("node exact vs hi-res truth (R % / P % / F1)")
    for g, row in ne.items():
        print(f"  {g:12s} " + "  ".join(f"{det}: {v['R']} / {v['P']} / {v['F1']}" for det, v in row.items()))
    for part, row in fibres().items():
        print(f"{part}: " + "  ".join(f"{det}: {v['R']} / {v['P']} / {v['F1']} (n {v['n']})" for det, v in row.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
