"""The `leave` and `step` alarms the detector uses as region support.

Same computation as Turncoat v3's primary operating point with only the surface prediction (no CT profiles, no other
segments, no growth prediction): `step` components on the calibrated uniform step score, and `leave` records with the
calibrated local support. Returns the alarm points in volume voxels (x, y, z).
"""
from __future__ import annotations

import time

import numpy as np

from ..tifxyz import Surface
from ..volume import Volume
from .fusion import Calibration, to_uniform
from .fusion3 import Calibration3, component_alarms
from .leave import scan
from .step import StepParams, step_channel


def leave_step_alarms(surface: Surface, voxel_um: float, prediction: Volume | None, *, scale: float = 1.0,
                      log=print) -> tuple[np.ndarray, dict]:
    """((A, 3) xyz of the primary `leave` and `step` alarms, info). `prediction` = the surface prediction volume (None:
    step alarms only)."""
    t0 = time.time()
    cal3, cal2 = Calibration3.load(), Calibration.load()
    pt = cal3.points[cal3.primary]
    step = step_channel(surface, StepParams(cal2.step_spans_vx, cal2.step_scales_vx))
    u_step = to_uniform(step, cal2)
    pts: list = []
    n_step = n_leave = 0
    if u_step is not None:
        al, _ = component_alarms(u_step, np.nan_to_num(u_step) > pt.step_u, surface.xyz, pt.step_min_nodes, "step")
        pts += [a.xyz for a in al]
        n_step = len(al)
    if prediction is not None:
        leave = scan(surface, prediction, scale)
        recs = leave.alarms(pt.a_support, cal3.a_radius_mm / (voxel_um * 1e-3))
        pts += [tuple(r[:3]) for r in recs]
        n_leave = len(recs)
    else:
        log("  leave alarms: not run (no surface prediction for this volume)")
    info = {"calibration": cal3.version, "point": cal3.primary, "step": n_step, "leave": n_leave,
            "seconds": round(time.time() - t0, 1)}
    return np.array(pts, float).reshape(-1, 3), info
