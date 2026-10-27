"""Which public surface prediction (m7) belongs to a CT volume? A registry of the open-data bucket's predictions.

`predictions.json` lists, per scroll, every surface prediction in the bucket (listed 2026-10-24) with the scan it
was made on (the leading timestamp of both names) and the pyramid level of the scan whose voxel frame it uses
(`-L0-` / `-L2-`). A mesh traced on level L of scan S can use a prediction of scan S at level L. m7 is preferred
over the recto models (it was the one calibrated in the bake-off).
"""
from __future__ import annotations

import json
import re
from importlib import resources

BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"


def registry() -> dict:
    return json.loads(resources.files("turncoat.alarms").joinpath("predictions.json").read_text())


def parse_volume(url: str) -> tuple[str | None, str | None, int]:
    """(scroll, scan id, level) from a volume URL such as
    .../PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/2 (level 2) or a level-0 root."""
    m = re.search(r"/(PHerc[^/]+)/volumes/(\d{14})-[^/]*?\.zarr(?:/(\d+))?/?$", url.rstrip("/"))
    if not m:
        return None, None, 0
    return m.group(1), m.group(2), int(m.group(3)) if m.group(3) else 0


def find_prediction(volume_url: str) -> tuple[str | None, str]:
    """(URL of the matching m7 surface prediction or None, explanation)."""
    scroll, scan, level = parse_volume(volume_url)
    if scroll is None:
        return None, "volume URL does not name an open-data scan; pass --prediction to use the instances channel"
    preds = registry()["scrolls"].get(scroll, [])
    match = [p for p in preds if p["scan"] == scan and p["level"] == level]
    if not match:
        have = ", ".join(f"{p['scan']} L{p['level']} {p['model']}" for p in preds) or "none"
        return None, f"no surface prediction for {scroll} scan {scan} level {level} (bucket has: {have})"
    match.sort(key=lambda p: (0 if "m7" in p["model"] else 1, p["model"]))
    return f"{BUCKET}/{match[0]['path']}", f"{match[0]['model']} for {scroll} scan {scan} level {level}"
