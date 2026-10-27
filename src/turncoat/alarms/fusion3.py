"""v3 alarms: an OR of per-channel rules, each at its own threshold. A channel that ran but found nothing simply
adds nothing (v2's mean vote let a silent relabel channel, u = 0.498 on 216 of 226 natural patches, cap the vote
below its threshold). Area floors only where a channel describes an area: step lines and the point channels have
none (the 2 mm2 floor deleted v2 step alarms within 1 mm of 94 of 155 natural switches).

  A        leave + alongside records with >= support records within 0.5 mm      -> one alarm per record
  step     8-connected components of u_step > t with >= min_nodes nodes          -> peak node + centroid node
  O        clusters of >= min_vertices vertices off the growth prediction (only when it is known)
                                                                                 -> centroid node
  relabel  switch regions of the relabelling (its own stability + corroboration rules), >= 2 mm2
                                                                                 -> peak node + centroid node
  cross    u_cross > t components >= 2 mm2 (plateau channel)                      -> peak node + centroid node

The thresholds live in calibration_v3.json (fitted on SwitchBench-natural, a development set for v3, and checked on
the PHerc0139 planted splices; PREREG_v3.md).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import numpy as np
from scipy.ndimage import label

MM_PER_VX_DEFAULT = 0.0096


class Calibration3Error(ValueError):
    """The v3 calibration file is missing or inconsistent."""


@dataclass(frozen=True)
class Point:
    name: str
    a_support: int
    step_u: float
    step_min_nodes: int
    o_min_vertices: int
    relabel_u: float
    cross_u: float
    plateau_min_mm2: float


@dataclass(frozen=True)
class Calibration3:
    version: str
    primary: str
    points: dict
    a_radius_mm: float
    notes: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Calibration3:
        try:
            text = Path(path).read_text() if path else \
                resources.files("turncoat.alarms").joinpath("calibration_v3.json").read_text()
            d = json.loads(text)
            pts = {k: Point(name=k, **v) for k, v in d["points"].items()}
            cal = cls(version=d["version"], primary=d["primary"], points=pts, a_radius_mm=float(d["a_radius_mm"]),
                      notes=d.get("notes", {}))
        except (OSError, KeyError, ValueError, TypeError) as exc:
            raise Calibration3Error(f"cannot load the v3 calibration: {exc}") from exc
        if cal.primary not in cal.points:
            raise Calibration3Error(f"primary point {cal.primary!r} missing")
        return cal


@dataclass(frozen=True)
class Alarm:
    channel: str
    xyz: tuple
    rc: tuple

    def as_dict(self) -> dict:
        return {"channel": self.channel, "xyz": [round(float(v), 2) for v in self.xyz], "rc": [int(v) for v in self.rc]}


def component_alarms(score: np.ndarray, mask: np.ndarray, xyz: np.ndarray, min_nodes: int, channel: str
                     ) -> tuple[list[Alarm], np.ndarray]:
    """Peak node and centroid node of each 8-connected component of `mask` with >= min_nodes nodes."""
    lab, n = label(mask, structure=np.ones((3, 3)))
    out, keep = [], np.zeros(mask.shape, bool)
    if n == 0:
        return out, keep
    sizes = np.bincount(lab.ravel())
    for k in range(1, n + 1):
        if sizes[k] < max(1, min_nodes):
            continue
        m = lab == k
        keep |= m
        rr, cc = np.nonzero(m)
        i = int(np.argmax(np.nan_to_num(score[rr, cc], nan=-np.inf)))
        j = int(np.argmin((rr - rr.mean()) ** 2 + (cc - cc.mean()) ** 2))
        for rc in dict.fromkeys([(int(rr[i]), int(cc[i])), (int(rr[j]), int(cc[j]))]):
            if np.isfinite(xyz[rc]).all():
                out.append(Alarm(channel, tuple(xyz[rc]), rc))
    return out, keep


def cluster_centroids(mask: np.ndarray, xyz: np.ndarray, min_nodes: int, channel: str) -> tuple[list[Alarm], np.ndarray]:
    lab, n = label(mask, structure=np.ones((3, 3)))
    out, keep = [], np.zeros(mask.shape, bool)
    for k in range(1, n + 1):
        rr, cc = np.nonzero(lab == k)
        if len(rr) < min_nodes:
            continue
        keep[rr, cc] = True
        j = int(np.argmin((rr - rr.mean()) ** 2 + (cc - cc.mean()) ** 2))
        if np.isfinite(xyz[rr[j], cc[j]]).all():
            out.append(Alarm(channel, tuple(xyz[rr[j], cc[j]]), (int(rr[j]), int(cc[j]))))
    return out, keep


def nodes_for_area(xyz: np.ndarray, mm2: float, mm_per_vx: float) -> int:
    d0 = np.linalg.norm(np.diff(xyz, axis=0), axis=-1)
    d1 = np.linalg.norm(np.diff(xyz, axis=1), axis=-1)
    a = float(np.nanmedian(d0) * np.nanmedian(d1)) * mm_per_vx ** 2
    return max(1, int(np.ceil(mm2 / max(a, 1e-12))))
