#!/usr/bin/env python3
"""Install a GaussMark watermarker into the evaluator's method directory.

    python3 method/export_watermarker.py                       # paper settings
    python3 method/export_watermarker.py --name my_variant --config '{"variance": 2e-5}'

Writes ``checkpoints/watermarkers/<name>/watermarker.py`` (a copy of
``method/gaussmark.py``) and ``config.json``.  The evaluators score every
directory under ``checkpoints/watermarkers/``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="gaussmark_200_tokens")
    ap.add_argument("--config", default="{}", help="JSON overrides of method/gaussmark.py DEFAULTS")
    ap.add_argument("--dest", default=str(ROOT / "checkpoints" / "watermarkers"))
    args = ap.parse_args()
    out = Path(args.dest) / args.name
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "method" / "gaussmark.py", out / "watermarker.py")
    (out / "config.json").write_text(json.dumps(json.loads(args.config), indent=2) + "\n")
    print(f"installed {out}")


if __name__ == "__main__":
    main()
