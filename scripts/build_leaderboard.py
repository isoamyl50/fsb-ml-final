#!/usr/bin/env python3
"""Rebuild experiments/leaderboard.md and experiments/leaderboard.csv from runs/*.json."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.leaderboard import build_leaderboard


def main() -> int:
    md = build_leaderboard()
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
