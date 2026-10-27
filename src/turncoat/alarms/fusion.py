"""Fusion: every channel on one scale (its CDF over clean PHerc0139 surface), a vote and an AND over the sheet-identity
channels, and the abrupt-step channel as a separate alarm.

u_c(node) = mid-rank CDF of channel c's score against its clean-surface reference sample (calibration file), so
u ~ 0.5 means "as ordinary as clean surface" and u -> 1 "rarer than any clean node".
  identity channels (instances, cross, relabel: they say WHICH sheet the node is on, so they see a slid plateau):
    mean (vote): the mean of u over the identity channels that have evidence at the node;
    min  (AND):  the minimum of u over those channels.
  step (the abrupt edge only): its own u threshold. On PHerc0139 dev plants, averaging it into the vote diluted
    it (cross+instances caught 12/13 plants, +step 7/13), so it is OR-ed instead: a node alarms when
    vote / t_vote > 1 or u_step / t_step > 1.
Every threshold was fixed on dev data per operating point (false alarms per cm2 of clean dev surface), and the vote
threshold per SET of identity channels that ran (a vote over fewer channels is noisier). Connected areas
(8-conn) of at least `min_area_mm2` are regions (PREREG_v2.md).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path

import numpy as np
from scipy import ndimage

from .channel import ChannelOutput, midrank_cdf


class CalibrationError(ValueError):
    """The calibration file is missing or inconsistent."""


@dataclass(frozen=True)
class Calibration:
    version: str
    references: dict           # channel -> sorted reference sample (np.ndarray)
    identity: tuple            # channels that enter the vote / AND
    rule: str                  # "mean" or "min": the primary rule
    primary: str               # key of the primary operating point
    points: dict               # op key -> {"fa_per_cm2", "step_u", "mean": {subset: thr}, "min": {subset: thr}}
    min_area_mm2: float
    step_scales_vx: tuple
    step_spans_vx: tuple
    notes: dict

    @classmethod
    def load(cls, path: str | Path | None = None) -> Calibration:
        try:
            text = Path(path).read_text() if path else \
                resources.files("turncoat.alarms").joinpath("calibration.json").read_text()
            d = json.loads(text)
            cal = cls(version=d["version"], references={k: np.asarray(v, np.float64) for k, v in d["references"].items()},
                      identity=tuple(d["identity_channels"]), rule=d["rule"], primary=str(d["primary"]),
                      points=d["operating_points"], min_area_mm2=float(d["min_area_mm2"]),
                      step_scales_vx=tuple(d["step"]["scales_vx"]), step_spans_vx=tuple(d["step"]["spans_vx"]),
                      notes=d.get("notes", {}))
        except (OSError, KeyError, ValueError, TypeError) as exc:
            raise CalibrationError(f"cannot load the v2 calibration: {exc}") from exc
        if cal.rule not in ("mean", "min") or cal.primary not in cal.points:
            raise CalibrationError("calibration names an unknown rule or operating point")
        missing = [c for c in (*cal.identity, "step") if c not in cal.references]
        if missing:
            raise CalibrationError(f"channels without a reference sample: {missing}")
        return cal

    def vote_threshold(self, point: str, rule: str, channels) -> float:
        key = subset_key(channels)
        try:
            return float(self.points[point][rule][key])
        except KeyError as exc:
            raise CalibrationError(f"no {rule} threshold for {key!r} at operating point {point}") from exc

    def step_threshold(self, point: str) -> float:
        return float(self.points[point]["step_u"])


def to_uniform(ch: ChannelOutput, cal: Calibration) -> np.ndarray | None:
    ref = cal.references.get(ch.name)
    return None if ref is None else midrank_cdf(ref, ch.score)


def fuse(us: list[np.ndarray], how: str) -> np.ndarray:
    if not us:
        raise ValueError("no channel to fuse")
    stack = np.stack(us)
    have = np.isfinite(stack).any(0)
    filled = np.where(np.isfinite(stack), stack, np.nan)
    with np.errstate(all="ignore"):
        if how == "mean":
            out = np.nanmean(np.where(have, filled, 0.0), axis=0)
        elif how == "min":
            out = np.nanmin(np.where(have, filled, 0.0), axis=0)
        else:
            raise ValueError(f"unknown fusion rule {how!r}")
    return np.where(have, out, np.nan)


def combined(vote: np.ndarray | None, step_u: np.ndarray | None, t_vote: float | None, t_step: float
             ) -> np.ndarray:
    """max(vote / t_vote, u_step / t_step): > 1 alarms (NaN where neither has evidence)."""
    parts = []
    if vote is not None and t_vote is not None:
        parts.append(vote / t_vote)
    if step_u is not None:
        parts.append(step_u / t_step)
    if not parts:
        raise ValueError("nothing to combine")
    out = parts[0]
    for p in parts[1:]:
        out = np.fmax(out, p)
    return out


def pool_max(a: np.ndarray, r: int) -> np.ndarray:
    """Max over a (2r+1)^2 window, NaN ignored (NaN only where the whole window is NaN). Boundary channels (step,
    instances) light the ramp of a switch and plateau channels (cross, relabel) its inside; pooling lets them meet."""
    if r <= 0:
        return a
    f = np.where(np.isfinite(a), a, -np.inf)
    m = ndimage.maximum_filter(f, size=2 * r + 1, mode="constant", cval=-np.inf)
    return np.where(np.isfinite(m), m, np.nan)


def subset_key(channels) -> str:
    return "+".join(sorted(channels))


@dataclass(frozen=True)
class FusedRegion:
    id: int
    nodes: int
    area_mm2: float
    peak: float                 # max of the combined map in the region (> 1)
    trigger: str                # "vote", "step" or "both": which alarm fired in the region
    rows: tuple[int, int]
    cols: tuple[int, int]
    peak_rc: tuple[int, int]
    centre_rc: tuple[int, int]  # region node nearest the region's centroid
    channels: dict              # channel -> share of the region's nodes where that channel alone has u > 0.95

    def as_dict(self) -> dict:
        return asdict(self)


def regions(fused: np.ndarray, threshold: float, node_area_mm2: float, min_area_mm2: float,
            us: dict[str, np.ndarray] | None = None, vote_hit: np.ndarray | None = None,
            step_hit: np.ndarray | None = None) -> tuple[list[FusedRegion], np.ndarray]:
    m = np.isfinite(fused) & (fused > threshold)
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    min_nodes = max(1, int(np.ceil(min_area_mm2 / max(node_area_mm2, 1e-12))))
    out, rid = [], np.zeros(fused.shape, np.int32)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    for k in range(1, n + 1):
        if sizes[k] < min_nodes:
            continue
        mk = lab == k
        rr, cc = np.nonzero(mk)
        i = int(np.argmax(fused[rr, cc]))
        j = int(np.argmin((rr - rr.mean()) ** 2 + (cc - cc.mean()) ** 2))
        share = {}
        for name, u in (us or {}).items():
            v = u[mk]
            share[name] = round(float(np.mean(v[np.isfinite(v)] > 0.95)), 3) if np.isfinite(v).any() else None
        v = bool(vote_hit[mk].any()) if vote_hit is not None else False
        st = bool(step_hit[mk].any()) if step_hit is not None else False
        trig = "both" if v and st else ("vote" if v else ("step" if st else "none"))
        reg = FusedRegion(id=len(out) + 1, nodes=int(sizes[k]), area_mm2=round(float(sizes[k] * node_area_mm2), 2),
                          peak=round(float(fused[rr[i], cc[i]]), 4), trigger=trig, rows=(int(rr.min()), int(rr.max()) + 1),
                          cols=(int(cc.min()), int(cc.max()) + 1), peak_rc=(int(rr[i]), int(cc[i])),
                          centre_rc=(int(rr[j]), int(cc[j])), channels=share)
        rid[mk] = reg.id
        out.append(reg)
    return out, rid


def alarm_points(regs: list[FusedRegion], xyz: np.ndarray) -> list[list[float]]:
    """Two alarm points per region (bake-off D rule, fixed before any bench score): its peak node and the region
    node nearest its centroid (one point when they coincide)."""
    pts = []
    for r in regs:
        for rc in dict.fromkeys([r.peak_rc, r.centre_rc]):
            p = xyz[rc]
            if np.isfinite(p).all():
                pts.append([round(float(v), 2) for v in p])
    return pts
