"""Command line: `turncoat check`, `turncoat embed` (GPU), `turncoat prefetch` (GPU machine helper)."""
from __future__ import annotations

import argparse
import sys

from . import __version__


def _check(argv: list[str]) -> int:
    from .check import CheckError, run_check
    from .embeddings import EmbeddingError
    from .tifxyz import TifxyzError
    from .volume import VolumeError
    ap = argparse.ArgumentParser(prog="turncoat check", description="Whole-segment sheet-switch review map.")
    ap.add_argument("segment", help="tifxyz segment folder (x.tif, y.tif, z.tif, meta.json)")
    ap.add_argument("--volume", help="the scan the segment was traced on (open-data zarr URL); picks the surface "
                    "prediction and the voxel size")
    ap.add_argument("--embeddings", help="dinovol embeddings from `turncoat embed` (default: <out>/embeddings.npz)")
    ap.add_argument("--out", help="output folder (default: <segment>/turncoat)")
    ap.add_argument("--prediction", help="surface prediction zarr URL, if the scan is not in the built-in registry")
    ap.add_argument("--no-prediction", action="store_true", help="skip the leave alarms (not the measured setup)")
    ap.add_argument("--voxel-um", type=float, help="voxel size in um (default: from the volume name, else 9.362)")
    ap.add_argument("--write-to-segment", action="store_true",
                    help="also write the maps into the segment folder for VC3D (ink-detections.json + channels)")
    ap.add_argument("--recompute-alarms", action="store_true", help="ignore the cached leave / step alarms")
    ap.add_argument("--memory-gb", type=float, default=0.6, help="chunk cache (GB) for the streamed prediction")
    a = ap.parse_args(argv)
    try:
        run_check(a.segment, volume=a.volume, embeddings=a.embeddings, out=a.out, prediction=a.prediction,
                  no_prediction=a.no_prediction, voxel_um=a.voxel_um, write_to_segment=a.write_to_segment,
                  recompute_alarms=a.recompute_alarms, memory_gb=a.memory_gb)
    except (CheckError, EmbeddingError, TifxyzError, VolumeError, OSError) as exc:
        print(f"turncoat check: {exc}", file=sys.stderr)
        return 2
    return 0


def _bench(argv: list[str]) -> int:
    import json

    from .bench import BenchError, load_bench, report, score, write_vc3d
    from .tifxyz import TifxyzError
    ap = argparse.ArgumentParser(prog="turncoat bench", description="Sheet-switch benchmark (labels in bench/).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("score", help="score a detector's per-node flags (one file per mesh) against the labels")
    sc.add_argument("--flags", required=True, help="folder with KEY.png / .npy / .npz / .tif per mesh (see bench/README.md)")
    sc.add_argument("--bench", help="benchmark folder (default: ./bench, else the one next to this package)")
    sc.add_argument("--split", choices=("test", "dev", "all"), default="test", help="meshes to score (default: test)")
    sc.add_argument("--meshes", help="comma-separated mesh keys, instead of a whole split")
    sc.add_argument("--json", help="also write the full result (per-mesh counts) to this file")
    vc = sub.add_parser("vc3d", help="write a mesh's labels into its downloaded tifxyz folder as VC3D overlay layers")
    vc.add_argument("segment", help="the published tifxyz folder of that mesh (see bench/manifest.json)")
    vc.add_argument("--mesh", required=True, help="mesh key, e.g. w034")
    vc.add_argument("--bench", help="benchmark folder (default: ./bench, else the one next to this package)")
    a = ap.parse_args(argv)
    try:
        b = load_bench(a.bench)
        if a.cmd == "vc3d":
            print("wrote " + ", ".join(write_vc3d(a.segment, a.mesh, b)))
            return 0
        meshes = [m.strip() for m in a.meshes.split(",") if m.strip()] if a.meshes else None
        res = score(a.flags, b, a.split, meshes)
    except (BenchError, TifxyzError, OSError) as exc:
        print(f"turncoat bench: {exc}", file=sys.stderr)
        return 2
    print(report(res))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(f"turncoat {__version__}\n\n"
              "  turncoat check SEGMENT --volume URL [--write-to-segment]   review map (CPU)\n"
              "  turncoat embed SEGMENT --volume URL --ckpt teacher.pt     dinovol embeddings (GPU only)\n"
              "  turncoat prefetch SEGMENT --volume URL --local DIR        copy the CT chunks embed needs (GPU box)\n"
              "  turncoat bench score --flags DIR                          score a detector on the benchmark\n"
              "\nRun `turncoat COMMAND --help` for the options.")
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd in ("--version", "version"):
        print(__version__)
        return 0
    if cmd == "check":
        return _check(rest)
    if cmd == "embed":
        from .gpu.embed_mesh import main as embed
        return embed(rest)
    if cmd == "prefetch":
        from .gpu.prefetch_ct import main as prefetch
        return prefetch(rest)
    if cmd == "bench":
        return _bench(rest)
    print(f"turncoat: unknown command {cmd!r} (check, embed, prefetch, bench)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
