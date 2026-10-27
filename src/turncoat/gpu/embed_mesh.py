"""GPU ONLY: dinovol embeddings along every node normal of a tifxyz mesh, in the file format `turncoat check` reads.
Refuses to run without CUDA (it is model inference; never on a laptop CPU).

Per valid node (or the --rows / --cols crop the check will use): t = -48..48 vx step 2 along the Turncoat node normal,
the dense head of scrollprize/dinovol_8um_ps192_102126 (64-d), global robust intensity normalisation, blended over the
fixed 192^3 window lattice (dinovol_channel.py). Output .npz: ok (H, W) bool, t (49,), dense (ok.sum(), 49, 64) int8
unit vectors x 127, crop (r0, r1, c0, c1). Cost on an RTX 3090 / 4090: ~0.3 s per 192^3 window; a whole PHerc0139
segment (~110k nodes) is a few thousand windows, roughly 15-30 min plus the CT reads.

Setup on the GPU machine (CUDA 12.x):
  pip install "torch==2.14.1" torchvision --index-url https://download.pytorch.org/whl/cu126
  git clone https://github.com/ScrollPrize/villa && pip install -e villa/dinovol      # dinovol_2
  pip install -e /path/to/turncoat
  curl -L -o teacher_step_640000.pt \
    https://huggingface.co/scrollprize/dinovol_8um_ps192_102126/resolve/main/teacher_step_640000.pt
usage: turncoat embed MESH --volume CT_URL[@level] --ckpt teacher_step_640000.pt [--out emb.npz]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

T_MAX, T_STEP = 48, 2
PCA_DIM = 16


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("--volume", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="", help="default: <segment>/turncoat/embeddings.npz")
    ap.add_argument("--rows", default=None)
    ap.add_argument("--cols", default=None)
    ap.add_argument("--batch", type=int, default=2)
    a = ap.parse_args(argv)
    if not a.out:
        a.out = str(Path(a.mesh) / "turncoat" / "embeddings.npz")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch
    except ImportError:
        print("turncoat embed: needs PyTorch with CUDA and the organisers' dinovol package (see the README, GPU setup)",
              file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("embed_mesh.py: no CUDA device; this is model inference and runs on a GPU machine only", file=sys.stderr)
        return 2
    from turncoat.gpu.dinovol_channel import (WIN, DinovolSampler, core_cell, embed_points, global_norm_stats, quantise,
                                              window_start)

    from turncoat.geometry import filled_normals
    from turncoat.util import crop
    from turncoat.tifxyz import read_tifxyz
    from turncoat.util import split_level as _split_level
    from turncoat.volume import Volume

    full = read_tifxyz(a.mesh)
    s = crop(full, a.rows, a.cols)
    r0, r1, c0, c1 = s.meta.get("turncoat_crop", [0, full.shape[0], 0, full.shape[1]])
    nrm = filled_normals(s)
    ok = s.valid & np.isfinite(nrm).all(-1)
    t = np.arange(-T_MAX, T_MAX + 1, T_STEP, dtype=np.float64)
    pts = s.xyz[ok][:, None, :] + t[None, :, None] * nrm[ok][:, None, :]
    zyx = pts.reshape(-1, 3)[:, ::-1]
    root, level = _split_level(a.volume)

    vol = Volume(root, level, memory_gb=4.0, workers=16)

    def read(lo, hi):
        return vol.read_box(lo, hi)

    sampler = DinovolSampler(a.ckpt)
    rng = np.random.default_rng(0)
    cores = np.unique(core_cell(zyx), axis=0)
    starts = window_start(cores[rng.choice(len(cores), min(32, len(cores)), replace=False)])
    raw = [read(st, st + WIN) for st in starts]
    norm = global_norm_stats(raw)
    # the patch tokens are kept only through a small PCA (the dense head is what the fallback reads); this bounds the
    # accumulator at nodes x offsets x PCA_DIM floats on whole meshes
    grids = [sampler.forward(np.stack([norm.apply(b)]).astype(np.float32))[0].float() for b in raw]
    sampler.fit_pca(grids, PCA_DIM)
    del grids, raw
    log(f"{int(ok.sum())} nodes x {len(t)} offsets, {len(cores)} windows; norm {norm}")
    emb = embed_points(sampler, read, zyx, norm, batch=a.batch, log=log)
    if "dense" not in emb:
        print("embed_mesh.py: this checkpoint has no dense head", file=sys.stderr)
        return 2
    dense = quantise(emb["dense"].reshape(int(ok.sum()), len(t), -1))
    np.savez(a.out, ok=ok, t=t, dense=dense, crop=np.array([r0, r1, c0, c1]))
    log(f"wrote {a.out}: dense {dense.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
