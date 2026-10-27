"""Small helpers: voxel size from a name, grid crops."""
from __future__ import annotations

import re

from .tifxyz import Surface


def guess_voxel_um(*names: str, default: float = 9.362) -> float:
    for n in names:
        m = re.search(r"(\d+(?:\.\d+)?)\s*um", str(n))
        if m:
            return float(m.group(1))
    return default


def parse_span(text: str | None, size: int) -> tuple[int, int]:
    """'a:b' grid span (either side may be empty) clipped to [0, size)."""
    if not text:
        return 0, size
    try:
        a, b = text.split(":")
        lo = int(a) if a.strip() else 0
        hi = int(b) if b.strip() else size
    except ValueError as exc:
        raise ValueError(f"bad span {text!r}, expected a:b") from exc
    lo, hi = max(0, lo), min(size, hi)
    if hi - lo < 3:
        raise ValueError(f"span {text!r} leaves fewer than 3 grid lines")
    return lo, hi


def crop(surface: Surface, rows: str | None, cols: str | None) -> Surface:
    """Grid crop of a surface (the report's grid indices are relative to the crop; the offset is kept in meta)."""
    if not rows and not cols:
        return surface
    r0, r1 = parse_span(rows, surface.shape[0])
    c0, c1 = parse_span(cols, surface.shape[1])
    meta = {**surface.meta, "turncoat_crop": [r0, r1, c0, c1]}
    return Surface(xyz=surface.xyz[r0:r1, c0:c1].copy(), valid=surface.valid[r0:r1, c0:c1].copy(), meta=meta,
                   path=surface.path)


def split_level(src: str) -> tuple[str, int]:
    """'URL[@level]' or a URL ending in /<level> -> (root, level)."""
    if "@" in src.rsplit("/", 1)[-1]:
        root, lvl = src.rsplit("@", 1)
        return root.rstrip("/"), int(lvl)
    s = src.rstrip("/")
    tail = s.rsplit("/", 1)[-1]
    if tail.isdigit():
        return s.rsplit("/", 1)[0], int(tail)
    return s, 0
