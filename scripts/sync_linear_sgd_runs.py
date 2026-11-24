"""Backfill existing approach 1 JSON logs into experiments/runs, without training."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.approach_1_linear_sgd.logging import sync_existing_run
from src.leaderboard import build_leaderboard


def main():
    paths = sorted((PROJECT_ROOT / "artifacts" / "approach_1_linear_sgd" / "runs").glob("*.json"))
    for path in paths:
        print(sync_existing_run(path, PROJECT_ROOT))
    build_leaderboard()
    print(f"Synchronized {len(paths)} runs; shared leaderboard refreshed. No model was trained or evaluated.")


if __name__ == "__main__":
    main()
