"""`turncoat check`: the whole-segment sheet-switch review map.

Inputs: a tifxyz segment, its dinovol embeddings (`turncoat embed`, GPU) and the scan it was traced on (to find the
organisers' public surface prediction for the `leave` alarms). Everything here is CPU array math plus streamed reads
of the surface prediction; nothing is downloaded to disk except the outputs.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from . import __version__
from . import detector as DT
from . import outputs as O
from .embeddings import dinovol_offset
from .geometry import node_area_mm2
from .tifxyz import Surface, read_tifxyz
from .util import guess_voxel_um

DEFAULT_OUT = "turncoat"            # <segment>/turncoat
EMBEDDINGS_NAME = "embeddings.npz"  # default embeddings file inside the output folder
ALARMS_NAME = "alarms.json"         # cached leave / step alarms inside the output folder


class CheckError(ValueError):
    pass


def default_out(segment: Path) -> Path:
    return Path(segment) / DEFAULT_OUT


def mesh_digest(s: Surface) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(np.nan_to_num(s.xyz, nan=-1.0)).astype(np.float32).tobytes())
    return h.hexdigest()[:16]


def resolve_prediction(volume: str | None, prediction: str | None, no_prediction: bool) -> tuple[str | None, str]:
    from .alarms.predictions import find_prediction
    if no_prediction:
        return None, "disabled (--no-prediction): step alarms only, not the measured configuration"
    if prediction:
        return prediction, "given"
    if not volume:
        raise CheckError("pass --volume (the scan the segment was traced on) or --prediction URL; "
                         "--no-prediction runs without leave alarms")
    url, why = find_prediction(volume)
    if url is None:
        raise CheckError(f"{why}. Pass --prediction URL, or --no-prediction")
    return url, why


def alarms_for(s: Surface, voxel_um: float, pred: str | None, cache: Path, *, recompute: bool, memory_gb: float,
               log=print) -> tuple[np.ndarray, dict]:
    """Leave / step alarm points, cached in `cache` per (mesh digest, prediction, voxel size)."""
    key = {"mesh": mesh_digest(s), "prediction": pred, "voxel_um": voxel_um}
    if cache.exists() and not recompute:
        try:
            d = json.loads(cache.read_text())
            if d.get("key") == key:
                log(f"  leave / step alarms: {len(d['xyz'])} reused from {cache}")
                return np.array(d["xyz"], float).reshape(-1, 3), {**d.get("info", {}), "cached": True}
        except (json.JSONDecodeError, KeyError, TypeError):
            log(f"  {cache} unreadable, recomputing")
    from .alarms.check import leave_step_alarms
    from .volume import Volume
    vol = Volume(pred, memory_gb=memory_gb) if pred else None
    log(f"  leave / step alarms: scanning the surface prediction along the mesh ({pred or 'none'})")
    xyz, info = leave_step_alarms(s, voxel_um, vol, log=log)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"key": key, "info": info, "xyz": [[round(float(v), 3) for v in p] for p in xyz]}))
    return xyz, {**info, "cached": False}


def alarm_hits(xyz: np.ndarray, valid: np.ndarray, alarms_xyz: np.ndarray, voxel_um: float) -> np.ndarray:
    """(H, W) bool: nodes within 1 mm of an alarm."""
    out = np.zeros(valid.shape, bool)
    if not len(alarms_xyz):
        return out
    G = np.where(valid[..., None], np.nan_to_num(xyz), 1e12).reshape(-1, 3)
    d, _ = cKDTree(alarms_xyz).query(G, distance_upper_bound=DT.V3_MM * 1000.0 / voxel_um)
    return (np.isfinite(d).reshape(valid.shape)) & valid


def run_check(segment: str | Path, *, volume: str | None = None, embeddings: str | Path | None = None,
              out: str | Path | None = None, prediction: str | None = None, no_prediction: bool = False,
              voxel_um: float | None = None, write_to_segment: bool = False, recompute_alarms: bool = False,
              memory_gb: float = 0.6, log=print) -> dict:
    t0 = time.time()
    segment = Path(segment)
    out = Path(out) if out else default_out(segment)
    emb = Path(embeddings) if embeddings else out / EMBEDDINGS_NAME
    if not emb.exists():
        raise CheckError(f"no embeddings at {emb}: run `turncoat embed {segment} --volume ...` on a GPU machine first "
                         "(or pass --embeddings FILE)")
    s = read_tifxyz(segment)
    vum = voxel_um or guess_voxel_um(volume or "", str(segment))
    pred, why = resolve_prediction(volume, prediction, no_prediction)
    log(f"turncoat {__version__}: {segment.name}, grid {s.shape}, {int(s.valid.sum())} nodes, {vum} um voxels")
    log(f"  surface prediction: {pred or 'none'} ({why})")
    t1 = time.time()
    off, ok, pitch = dinovol_offset(emb, s.shape)
    log(f"  dinovol offsets: {int(ok.sum())} embedded nodes, layer pitch {pitch:.2f} vx ({time.time() - t1:.0f} s)")
    al_xyz, al_info = alarms_for(s, vum, pred, out / ALARMS_NAME, recompute=recompute_alarms, memory_gb=memory_gb,
                                 log=log)
    flagged = DT.flags(off, pitch, ok, s.xyz, s.valid, vum, al_xyz)
    lk = DT.likelihood(off, pitch, ok, s.valid)
    maps = {"switch_likelihood": lk, "suspicion": O.suspicion_map(flagged, ok, s.valid)}
    O.write_maps(out, maps)
    layers = np.where(ok & s.valid, off / pitch, np.nan)
    regs = O.regions(flagged, layers, lk, s.xyz, node_area_mm2(s, vum), alarm_hits(s.xyz, s.valid, al_xyz, vum))
    O.write_regions(out, regs, segment.name)
    summary = {"turncoat": __version__, "segment": str(segment), "grid": list(s.shape), "voxel_um": vum,
               "prediction": pred, "embeddings": str(emb), "pitch_vx": round(float(pitch), 3),
               "rule": {"a2_min": DT.A2_MIN, "min_mm2": DT.MIN_MM2, "alarms_min": DT.V3_MIN, "alarms_mm": DT.V3_MM},
               "nodes": {"valid": int(s.valid.sum()), "embedded": int((ok & s.valid).sum()),
                         "flagged": int(flagged.sum())},
               "alarms": {**al_info, "points": int(len(al_xyz))}, "regions": len(regs),
               "seconds": round(time.time() - t0, 1)}
    if write_to_segment:
        summary["written_to_segment"] = O.write_to_segment(segment, maps, s.meta)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    share = flagged.sum() / max(int((ok & s.valid).sum()), 1)
    log(f"  {len(regs)} region(s), {int(flagged.sum())} flagged nodes ({share:.1%} of embedded); outputs in {out}")
    for r in regs[:5]:
        log(f"    #{r.id}: {r.area_mm2:.1f} mm2, likelihood {r.mean_likelihood:.2f}, {r.verdict}, at xyz {r.worst_xyz}")
    return summary
