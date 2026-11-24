"""Compatibility entry point; canonical CLI remains scripts/run_experiment.py."""

from scripts.run_experiment import main

if __name__ == "__main__":
    raise SystemExit(main())
