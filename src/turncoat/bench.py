"""Sheet-switch benchmark: score any detector's per-node flags against the published labels in `bench/`.

Rules (the ones behind the README's Evidence table):
- node exact: on each mesh, tp = flagged label-0 (off-sheet) nodes, fp = flagged label-1 (on-sheet) nodes, positives =
  label-0 nodes; unlabelled nodes (255) are not scored. Pooled = summed counts over the meshes of a group.
- fibre pairs: a detector calls a pair "two sheets" when exactly one of its two nodes is flagged; positives = pairs the
  fibres put on two sheets ("different"); tp / fp = called pairs that are different / same.
- verified nodes: node exact, restricted to the nodes where the truth and the fibres agree.
- human-fibre nodes: node exact against the labels from human-guided fibres alone (PHerc0139).
R = tp / positives, P = tp / (tp + fp), F1 = 2PR / (P + R); 0 when tp = 0.

Flags: one file per mesh in the flags folder, named after the mesh key (see bench/manifest.json), any of
  KEY.png (8-bit, nonzero = flagged), KEY.npy (bool / int, nonzero = flagged), KEY.npz (array "flags"),
  KEY.tif (float, > 0.5 = flagged; NaN = not flagged), or a folder KEY/ holding Turncoat's suspicion.tif or
  turncoat_flagged.png. The array must have the segment's tifxyz grid shape (rows, cols).
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

UNLABELLED, OFF, ON = 255, 0, 1
GROUP_ORDER = ("test", "PHerc0139", "PHerc0500P2", "PHerc0814", "PHerc0841", "dev")


class BenchError(ValueError):
    """The benchmark folder or a flag file is missing or malformed."""


@dataclass(frozen=True)
class Bench:
    root: Path
    manifest: dict
    labels: dict           # key -> (H, W) uint8 (1 on, 0 off, 255 unlabelled)
    pairs: dict            # key -> ((N, 4) int rows r_a, c_a, r_b, c_b; (N,) bool different)
    verified: dict         # key -> (N, 2) int rows, cols
    human: dict            # key -> ((N, 2) int rows, cols; (N,) uint8 label)

    def keys(self, split: str = "test") -> list[str]:
        return [k for k, v in self.manifest.items() if split == "all" or v["split"] == split]


def default_root() -> Path:
    here = Path.cwd() / "bench"
    if (here / "manifest.json").exists():
        return here
    return Path(__file__).resolve().parents[2] / "bench"


def _read_png(path: Path) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as im:
        a = np.asarray(im)
    return a if a.ndim == 2 else a[..., :3].max(-1)


def _csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise BenchError(f"missing {path}")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _group_rc(rows: list[dict], cols: tuple[str, ...]) -> dict:
    out: dict = {}
    for r in rows:
        out.setdefault(r["mesh"], []).append([int(r[c]) for c in cols])
    return {k: np.array(v, np.int64).reshape(-1, len(cols)) for k, v in out.items()}


def load_bench(root: Path | str | None = None) -> Bench:
    root = Path(root) if root else default_root()
    mf = root / "manifest.json"
    if not mf.exists():
        raise BenchError(f"{root} is not a benchmark folder (no manifest.json)")
    manifest = json.loads(mf.read_text())
    labels = {}
    for k, v in manifest.items():
        p = root / "labels" / f"{k}.png"
        if not p.exists():
            raise BenchError(f"missing label grid {p}")
        g = _read_png(p)
        if list(g.shape) != v["grid_shape"] or not np.isin(g, (OFF, ON, UNLABELLED)).all():
            raise BenchError(f"{p}: expected {v['grid_shape']} with values 0 / 1 / 255")
        labels[k] = g
    prow = _csv_rows(root / "fibre_pairs.csv")
    rel = {}
    for r in prow:
        if r["relation"] not in ("same", "different"):
            raise BenchError(f"fibre_pairs.csv: bad relation {r['relation']!r}")
        rel.setdefault(r["mesh"], []).append(r["relation"] == "different")
    rc = _group_rc(prow, ("row_a", "col_a", "row_b", "col_b"))
    pairs = {k: (rc[k], np.array(rel[k], bool)) for k in rc}
    verified = _group_rc(_csv_rows(root / "verified_nodes.csv"), ("row", "col"))
    hrows = _csv_rows(root / "human_fibre_nodes.csv")
    hrc = _group_rc(hrows, ("row", "col", "label"))
    human = {k: (v[:, :2], v[:, 2].astype(np.uint8)) for k, v in hrc.items()}
    return Bench(root, manifest, labels, pairs, verified, human)


def _flag_file(flags_dir: Path, key: str) -> Path | None:
    for name in (f"{key}.png", f"{key}.npy", f"{key}.npz", f"{key}.tif", f"{key}.tiff",
                 f"{key}/suspicion.tif", f"{key}/turncoat_flagged.png"):
        if (flags_dir / name).exists():
            return flags_dir / name
    return None


def read_flags(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """One mesh's flags as a bool (H, W) array."""
    suf = path.suffix.lower()
    try:
        if suf == ".png":
            a = _read_png(path) > 0
        elif suf == ".npy":
            a = np.load(path, allow_pickle=False) != 0
        elif suf == ".npz":
            with np.load(path, allow_pickle=False) as z:
                if "flags" not in z.files:
                    raise BenchError(f"{path}: no array named 'flags'")
                a = z["flags"] != 0
        elif suf in (".tif", ".tiff"):
            import tifffile
            a = np.nan_to_num(np.asarray(tifffile.imread(path), float), nan=0.0) > 0.5
        else:
            raise BenchError(f"{path}: unsupported flag file type")
    except (OSError, ValueError) as exc:
        if isinstance(exc, BenchError):
            raise
        raise BenchError(f"{path}: cannot read flags ({exc})") from exc
    if a.shape != tuple(shape):
        raise BenchError(f"{path}: flags are {a.shape}, the mesh grid is {tuple(shape)}")
    return a


def prf(tp: int, fp: int, pos: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / pos if pos else 0.0
    return {"R": round(100 * r, 1), "P": round(100 * p, 1), "F1": round(2 * p * r / (p + r), 3) if tp else 0.0,
            "tp": int(tp), "fp": int(fp), "pos": int(pos)}


def node_counts(flags: np.ndarray, labels: np.ndarray) -> dict:
    off, on = labels == OFF, labels == ON
    return {"tp": int((flags & off).sum()), "fp": int((flags & on).sum()), "off": int(off.sum()), "on": int(on.sum())}


def pair_counts(flags: np.ndarray, rc: np.ndarray, different: np.ndarray) -> dict:
    called = flags[rc[:, 0], rc[:, 1]] != flags[rc[:, 2], rc[:, 3]]
    return {"tp": int((called & different).sum()), "fp": int((called & ~different).sum()), "pos": int(different.sum())}


def _sub_counts(flags: np.ndarray, rc: np.ndarray, lab: np.ndarray) -> dict:
    f = flags[rc[:, 0], rc[:, 1]]
    return {"tp": int((f & (lab == OFF)).sum()), "fp": int((f & (lab == ON)).sum()), "pos": int((lab == OFF).sum())}


def _mesh_scores(b: Bench, key: str, flags: np.ndarray) -> dict:
    lab = b.labels[key]
    out = {"node_exact": node_counts(flags, lab)}
    if key in b.pairs:
        out["fibre_pairs"] = pair_counts(flags, *b.pairs[key])
    if key in b.verified:
        v = b.verified[key]
        out["verified_nodes"] = _sub_counts(flags, v, lab[v[:, 0], v[:, 1]])
    if key in b.human:
        rc, hl = b.human[key]
        out["human_fibre_nodes"] = _sub_counts(flags, rc, hl)
    return out


def _pool(rows: list[dict], part: str) -> dict | None:
    rows = [r[part] for r in rows if part in r]
    if not rows:
        return None
    tp, fp = sum(r["tp"] for r in rows), sum(r["fp"] for r in rows)
    pos = sum(r["off"] if "off" in r else r["pos"] for r in rows)
    return prf(tp, fp, pos)


def score(flags_dir: Path | str, bench: Bench | None = None, split: str = "test",
          meshes: list[str] | None = None) -> dict:
    """Per-mesh counts and pooled R / P / F1 for the meshes of a split ('test', 'dev' or 'all')."""
    b = bench or load_bench()
    flags_dir = Path(flags_dir)
    if not flags_dir.is_dir():
        raise BenchError(f"{flags_dir} is not a folder")
    keys = meshes or b.keys(split)
    unknown = [k for k in keys if k not in b.manifest]
    if unknown:
        raise BenchError(f"unknown mesh keys: {', '.join(unknown)}")
    files = {k: _flag_file(flags_dir, k) for k in keys}
    missing = [k for k, p in files.items() if p is None]
    if missing:
        raise BenchError(f"no flags in {flags_dir} for: {', '.join(missing)} (pass --meshes to score a subset)")
    per = {k: _mesh_scores(b, k, read_flags(files[k], b.labels[k].shape)) for k in keys}
    groups: dict = {}
    for k in keys:
        m = b.manifest[k]
        for g in ((m["split"], m["scroll"]) if m["split"] == "test" else (m["split"],)):
            groups.setdefault(g, []).append(k)
    pooled = {}
    for g in sorted(groups, key=lambda x: GROUP_ORDER.index(x) if x in GROUP_ORDER else len(GROUP_ORDER)):
        rows = [per[k] for k in groups[g]]
        pooled[g] = {"meshes": groups[g], **{part: s for part in ("node_exact", "fibre_pairs", "verified_nodes",
                                                                    "human_fibre_nodes") if (s := _pool(rows, part))}}
    rates = {k: prf(v["node_exact"]["tp"], v["node_exact"]["fp"], v["node_exact"]["off"])["F1"] for k, v in per.items()}
    return {"flags": str(flags_dir), "per_mesh": per, "per_mesh_node_exact_f1": rates, "pooled": pooled}


def report(res: dict) -> str:
    lines = ["group           part                R %     P %     F1      tp / pos"]
    for g, row in res["pooled"].items():
        for part in ("node_exact", "fibre_pairs", "verified_nodes", "human_fibre_nodes"):
            if part in row:
                s = row[part]
                lines.append(f"{g:15s} {part:18s} {s['R']:6.1f}  {s['P']:6.1f}  {s['F1']:.3f}   {s['tp']} / {s['pos']}")
    return "\n".join(lines)


VC3D_FILES = {"bench_truth_off.png": "Benchmark truth: off sheet", "bench_truth_unlabelled.png": "Benchmark truth: unlabelled"}


def write_vc3d(segment: Path | str, key: str, bench: Bench | None = None) -> list[str]:
    """Write the mesh's labels into a downloaded copy of its tifxyz folder as two VC3D Ink Detection layers."""
    from PIL import Image

    from .outputs import FLAG_PNG_VALUE, _merge_ink_index
    from .tifxyz import read_tifxyz
    b = bench or load_bench()
    if key not in b.manifest:
        raise BenchError(f"unknown mesh key {key!r} (one of {', '.join(b.manifest)})")
    seg = Path(segment)
    s = read_tifxyz(seg)
    lab = b.labels[key]
    if s.shape != lab.shape:
        raise BenchError(f"{seg} has grid {s.shape}; {key} labels are {lab.shape}: not the published segment")
    import hashlib
    if hashlib.sha256((seg / "x.tif").read_bytes()).hexdigest() != b.manifest[key]["x_tif_sha256"]:
        print(f"warning: {seg}/x.tif differs from the published file the labels were made on")
    maps = {"bench_truth_off.png": lab == OFF, "bench_truth_unlabelled.png": lab == UNLABELLED}
    entries = []
    for fn, m in maps.items():
        Image.fromarray(np.where(m, FLAG_PNG_VALUE, 0).astype(np.uint8)).save(seg / fn)
        entries.append({"label": VC3D_FILES[fn], "segment_id": seg.name, "segment_long_id": str(s.meta.get("uuid") or seg.name),
                        "artifact_type": "turncoat", "local_file": fn})
    _merge_ink_index(seg, entries)
    return [*maps, "ink-detections.json"]
