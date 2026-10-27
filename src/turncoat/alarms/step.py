"""`step` channel: coherent normal steps in the mesh geometry, from the mesh alone (no CT).

Idea credited to TIFXYZ Doctor's `coherent-normal-step` cue (aviad12g/tifxyz-doctor, MIT): a grid edge whose
vector has a large component along the surface normal is a step in the surface, and a step that runs along a
coherent band is where a trace jumped to another sheet. This is an independent implementation (no code copied)
with two changes aimed at what that cue misses (gradual transitions, 5 % caught on the switch-truth bench). On
PHerc0139 dev data the long spans did NOT rescue gradual slides (clean |projection| grows to 37 / 99 vx at 160 /
320 vx: papyrus is wavy at mm scales), so v2 ships spans 20 / 40 / 80 vx and uses this channel for abrupt steps:

* Multi-scale spans. For span s (in grid nodes) the node's chord p[i + s - h] - p[i - h] (h = s // 2) is projected
  on the bisector of the two END normals. On any circular arc that projection is exactly zero, so the bending of
  the scroll does not register; a step of height d inside the span gives |projection| ~ d. Measured with the
  mesh's own normals at every node instead (span 1), a gradual ramp tilts the normals with it and disappears;
  with the end normals of a span longer than the ramp it does not.
* Per-scale normalisation. Each span's |projection| is divided by a clean-surface scale (calibrated on PHerc0139
  clean meshes and stored in the v2 calibration), and the node score is the maximum over spans and both grid axes.

Spans are given in voxels and converted to nodes with the mesh's grid step, so a 20 vx grid and a 10 vx grid see
the same physical scales.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import normals
from ..tifxyz import Surface
from .channel import ChannelOutput

DEFAULT_SPANS_VX = (20.0, 40.0, 80.0, 160.0, 320.0)


@dataclass(frozen=True)
class StepParams:
    spans_vx: tuple[float, ...] = DEFAULT_SPANS_VX
    # clean-surface scale (vx) of |projection| per span; None -> unscaled (calibration run only)
    scales_vx: tuple[float, ...] | None = None


def _shift(a: np.ndarray, k: int, axis: int) -> np.ndarray:
    """out[i] = a[i + k] along `axis`, NaN where i + k is outside the grid (no wrap-around)."""
    out = np.full_like(a, np.nan)
    n = a.shape[axis]
    if abs(k) >= n:
        return out
    src = [slice(None)] * a.ndim
    dst = [slice(None)] * a.ndim
    if k >= 0:
        src[axis], dst[axis] = slice(k, n), slice(0, n - k)
    else:
        src[axis], dst[axis] = slice(0, n + k), slice(-k, n)
    out[tuple(dst)] = a[tuple(src)]
    return out


def span_projection(xyz: np.ndarray, nrm: np.ndarray, span: int, axis: int) -> np.ndarray:
    """(H, W) signed chord projection on the bisector of the end normals, centred on each node (NaN at borders)."""
    h = span // 2
    a, b = _shift(xyz, -h, axis), _shift(xyz, span - h, axis)
    na, nb = _shift(nrm, -h, axis), _shift(nrm, span - h, axis)
    bis = na + nb
    ln = np.linalg.norm(bis, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        bis = bis / ln
    d = np.einsum("...k,...k->...", b - a, bis)
    if span == 1:   # an edge belongs to both of its ends
        d = np.where(np.abs(np.nan_to_num(d)) >= np.abs(np.nan_to_num(_shift(d, -1, axis))), d, _shift(d, -1, axis))
    return d


def spans_in_nodes(surface: Surface, spans_vx: tuple[float, ...]) -> tuple[int, ...]:
    """Each span in grid nodes (>= 1) on this mesh's grid; spans that round to the same node count stay listed."""
    step = surface.grid_step()
    return tuple(max(1, int(round(s / max(step, 1e-6)))) for s in spans_vx)


def step_fields(surface: Surface, spans_vx: tuple[float, ...] = DEFAULT_SPANS_VX) -> np.ndarray:
    """(n_spans, H, W) max over both axes of |projection| per span (vx); NaN where neither axis has a value. On a
    coarse grid two spans can round to the same node count: the field is then computed once and repeated."""
    nrm, ok = normals(surface.xyz, surface.valid)
    xyz = np.where(ok[..., None], surface.xyz, np.nan)
    out = np.full((len(spans_vx), *surface.shape), np.nan)
    done: dict[int, np.ndarray] = {}
    for i, s in enumerate(spans_in_nodes(surface, spans_vx)):
        if s not in done:
            per_axis = [np.abs(span_projection(xyz, nrm, s, ax)) for ax in (0, 1)]
            done[s] = np.fmax(per_axis[0], per_axis[1])
        out[i] = done[s]
    return out


def step_channel(surface: Surface, params: StepParams) -> ChannelOutput:
    if params.scales_vx is None:
        raise ValueError("step channel needs calibrated per-span scales (StepParams.scales_vx)")
    if len(params.scales_vx) != len(params.spans_vx):
        raise ValueError("one scale per span is required")
    f = step_fields(surface, params.spans_vx)
    z = f / np.asarray(params.scales_vx, dtype=np.float64)[:, None, None]
    with np.errstate(all="ignore"):
        score = np.nanmax(np.where(np.isfinite(z), z, -np.inf), axis=0)
        best = np.nanargmax(np.where(np.isfinite(z), z, -np.inf), axis=0)
    score = np.where(np.isfinite(score), score, np.nan)
    spans = np.asarray(params.spans_vx)[best]
    return ChannelOutput("step", score, None, {"spans_vx": list(params.spans_vx), "scales_vx": list(params.scales_vx),
                                               "grid_step_vx": round(surface.grid_step(), 2),
                                               "best_span_vx_median": float(np.nanmedian(np.where(np.isfinite(score),
                                                                                                    spans, np.nan)))
                                               if np.isfinite(score).any() else None})
