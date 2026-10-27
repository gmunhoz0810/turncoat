"""Score a detector's per-node flags against the benchmark labels (same as `turncoat bench score`).

  python bench/score_detector.py --flags DIR [--split test|dev|all] [--meshes w034,p5a] [--json OUT]
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    from turncoat.cli import main
except ImportError:   # run from a clone without installing
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from turncoat.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["bench", "score", *sys.argv[1:]]))
