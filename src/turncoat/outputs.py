"""Output files of `turncoat check`.

In the output folder:
  regions.json            ranked candidate regions (review the first ones first), one object per region
  turncoat_regions.json   the same regions as a VC3D point collection (one point per region, volume voxels)
  switch_likelihood.tif   float32 on the mesh grid, 0-1 (1.0 at half a layer off the dominant sheet), NaN = no data
  suspicion.tif           float32 on the mesh grid: 1.0 flagged, 0.0 checked and clean, NaN = no data
  summary.json            counts, timings and the inputs used

With --write-to-segment, also into the tifxyz folder:
  switch_likelihood.tif, suspicion.tif     surface channels (VC3D builds that list a segment's own channels)
  turncoat_switch_likelihood.png, turncoat_flagged.png, ink-detections.json
      8-bit maps on the mesh grid plus their entries in ink-detections.json, which VC3D's Ink Detection overlay reads,
      so any recent VC3D shows the map over the segment without a plugin. Entries made by other tools are kept.
It never writes x/y/z.tif, meta.json, mask.tif, approval.tif or corrections.json.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

PROTECTED = {"x.tif", "y.tif", "z.tif", "mask.tif", "approval.tif", "corrections.json", "meta.json"}
INK_INDEX = "ink-detections.json"
INK_LABELS = {"turncoat_switch_likelihood.png": "Turncoat switch likelihood",
              "turncoat_flagged.png": "Turncoat flagged regions"}
COLLECTION_COLOUR = [1.0, 0.24, 0.96]


@dataclass(frozen=True)
class Region:
    id: int
    nodes: int
    area_mm2: float
    layers: float                # median signed dinovol offset of the region, in layers (+ along the normal)
    mean_likelihood: float
    score: float                 # review rank key: area_mm2 x mean_likelihood
    alarms_within_1mm: int
    verdict: str
    rows: tuple[int, int]
    cols: tuple[int, int]
    worst_rc: tuple[int, int]
    worst_xyz: list
    centroid_xyz: list

    def as_dict(self) -> dict:
        d = asdict(self)
        return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in d.items()}


def _round2(v) -> list:
    return [round(float(x), 2) for x in v]


def suspicion_map(flagged: np.ndarray, ok: np.ndarray, valid: np.ndarray) -> np.ndarray:
    s = np.full(valid.shape, np.nan, np.float32)
    s[ok & valid] = 0.0
    s[flagged & valid] = 1.0
    return s


def regions(flagged: np.ndarray, layers: np.ndarray, lk: np.ndarray, xyz: np.ndarray, node_area_mm2: float,
            alarm_hits: np.ndarray) -> list[Region]:
    """One region per 8-connected flagged component, ranked by area x mean likelihood (id 1 = review first)."""
    lab, _ = ndimage.label(flagged, structure=np.ones((3, 3)))
    out: list[Region] = []
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        rr, cc = np.nonzero(lab[sl] == k)
        rr, cc = rr + sl[0].start, cc + sl[1].start
        lay = layers[rr, cc]
        lay = lay[np.isfinite(lay)]
        med = float(np.median(lay)) if len(lay) else float("nan")
        mag = np.abs(np.nan_to_num(layers[rr, cc]))
        i = int(np.argmax(mag))
        lkv = lk[rr, cc]
        mlk = float(np.nanmean(lkv)) if np.isfinite(lkv).any() else 0.0
        area = round(len(rr) * node_area_mm2, 3)
        side = "along" if med > 0 else "against"
        verdict = (f"about {abs(med):.1f} layer(s) off the segment's main sheet, {side} its normal"
                   if np.isfinite(med) else "off the segment's main sheet")
        out.append(Region(0, int(len(rr)), area, round(med, 3), round(mlk, 3), round(area * mlk, 3),
                          int(alarm_hits[rr, cc].sum()), verdict, (int(rr.min()), int(rr.max()) + 1),
                          (int(cc.min()), int(cc.max()) + 1), (int(rr[i]), int(cc[i])), _round2(xyz[rr[i], cc[i]]),
                          _round2(np.nanmean(xyz[rr, cc], 0))))
    out.sort(key=lambda r: -r.score)
    return [replace(r, id=k) for k, r in enumerate(out, start=1)]


def _json_default(v):
    f = float(v)
    return None if not np.isfinite(f) else f


def write_regions(out: Path, regs: list[Region], mesh_name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "regions.json").write_text(json.dumps([r.as_dict() for r in regs], indent=1, default=_json_default))
    now = int(time.time())
    pts = {str(r.id): {"p": r.worst_xyz, "creation_time": now, "wind_a": None} for r in regs
           if np.isfinite(r.worst_xyz).all()}
    coll = {"uuid": str(uuid.uuid4()), "name": f"turncoat {mesh_name}", "points": pts,
            "metadata": {"winding_is_absolute": False}, "color": COLLECTION_COLOUR}
    pc = {"vc_pointcollections_json_version": "1", "collections": {"1": coll},
          "turncoat": {"regions": [r.as_dict() for r in regs]}}
    (out / "turncoat_regions.json").write_text(json.dumps(pc, indent=1, default=_json_default))


def write_maps(out: Path, maps: dict[str, np.ndarray]) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    for name, a in maps.items():
        tifffile.imwrite(out / f"{name}.tif", a.astype(np.float32))
    return [f"{n}.tif" for n in maps]


def to_u8(a: np.ndarray) -> np.ndarray:
    """0-1 float map -> uint8 (0 = transparent in VC3D's overlay; NaN -> 0)."""
    return np.round(np.clip(np.nan_to_num(a, nan=0.0), 0.0, 1.0) * 255.0).astype(np.uint8)


def _merge_ink_index(seg_dir: Path, entries: list[dict]) -> None:
    path = seg_dir / INK_INDEX
    old: list = []
    if path.exists():
        try:
            old = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} is not valid JSON; not touching it") from exc
        if not isinstance(old, list):
            raise ValueError(f"{path} is not a JSON array; not touching it")
    ours = {e["local_file"] for e in entries}
    kept = [e for e in old if not (isinstance(e, dict) and e.get("local_file") in ours)]
    path.write_text(json.dumps(kept + entries, indent=1))


def write_to_segment(seg_dir: Path, maps: dict[str, np.ndarray], meta: dict) -> list[str]:
    """Write the float channels, the 8-bit overlay images and their ink-detections.json entries into the segment."""
    from PIL import Image
    seg_dir = Path(seg_dir)
    if not (seg_dir / "x.tif").exists():
        raise ValueError(f"{seg_dir} is not a tifxyz folder (no x.tif)")
    written = []
    for name, a in maps.items():
        fn = f"{name}.tif"
        if fn in PROTECTED:
            raise ValueError(f"refusing to write {fn} into a segment")
        tifffile.imwrite(seg_dir / fn, a.astype(np.float32))
        written.append(fn)
    images = {"turncoat_switch_likelihood.png": maps["switch_likelihood"],
              "turncoat_flagged.png": np.where(maps["suspicion"] >= 1.0, 1.0, 0.0)}
    seg_id = seg_dir.name
    long_id = str(meta.get("uuid") or seg_id)
    entries = []
    for fn, a in images.items():
        Image.fromarray(to_u8(a)).save(seg_dir / fn)
        written.append(fn)
        entries.append({"label": INK_LABELS[fn], "segment_id": seg_id, "segment_long_id": long_id,
                        "artifact_type": "turncoat", "local_file": fn})
    _merge_ink_index(seg_dir, entries)
    written.append(INK_INDEX)
    return written
