"""GPU-machine helper: copy every CT chunk the dinovol windows of a mesh need into a local Zarr level (same chunk files
as published), so embed_mesh.py reads from disk instead of re-fetching overlapping windows over HTTP (the GPU sat idle
on HTTP reads otherwise).
usage: turncoat prefetch MESH --volume URL[@level] --local DIR [--threads 64]   then turncoat embed --volume DIR@level
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

T_MAX, T_STEP = 48, 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("--volume", required=True)
    ap.add_argument("--local", required=True)
    ap.add_argument("--threads", type=int, default=64)
    a = ap.parse_args(argv)
    from turncoat.gpu.dinovol_channel import WIN, core_cell, window_start

    from turncoat.geometry import filled_normals
    from turncoat.tifxyz import read_tifxyz
    from turncoat.util import split_level as _split_level
    from turncoat.volume import Volume

    s = read_tifxyz(a.mesh)
    nrm = filled_normals(s)
    ok = s.valid & np.isfinite(nrm).all(-1)
    t = np.arange(-T_MAX, T_MAX + 1, T_STEP, dtype=np.float64)
    zyx = (s.xyz[ok][:, None, :] + t[None, :, None] * nrm[ok][:, None, :]).reshape(-1, 3)[:, ::-1]
    starts = window_start(np.unique(core_cell(zyx), axis=0))
    root, level = _split_level(a.volume)
    vol = Volume(root, level, memory_gb=0.1)
    keys = set()
    for st in starts:
        rngs = vol._chunk_ranges(st, st + WIN)
        if rngs is not None:
            keys.update((i, j, k) for i in rngs[0] for j in rngs[1] for k in rngs[2])
    dst = Path(a.local) / str(level)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / ".zarray").write_text(json.dumps(vol.meta))

    def get(key):
        rel = vol._key(key)
        f = dst / rel
        if f.exists():
            return "have"
        buf = vol.store.get(rel)
        if buf is None:
            return "absent"
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_name(f.name + f".{os.getpid()}.tmp")
        tmp.write_bytes(buf)
        os.replace(tmp, f)
        return "got"

    with ThreadPoolExecutor(a.threads) as ex:
        res = list(ex.map(get, sorted(keys)))
    print(json.dumps({"mesh": a.mesh, "windows": int(len(starts)), "chunks": len(keys),
                      **{k: res.count(k) for k in ("got", "have", "absent")}}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
