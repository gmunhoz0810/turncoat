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


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(f"turncoat {__version__}\n\n"
              "  turncoat check SEGMENT --volume URL [--write-to-segment]   review map (CPU)\n"
              "  turncoat embed SEGMENT --volume URL --ckpt teacher.pt     dinovol embeddings (GPU only)\n"
              "  turncoat prefetch SEGMENT --volume URL --local DIR        copy the CT chunks embed needs (GPU box)\n"
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
    print(f"turncoat: unknown command {cmd!r} (check, embed, prefetch)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
