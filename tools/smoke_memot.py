"""Тонкая точка входа для MeMOT smoke-suite."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_detection.smoke import run  # noqa: E402

if __name__ == "__main__":
    config = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    run(config)
