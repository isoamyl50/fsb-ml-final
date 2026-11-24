"""Verify the six real-data runs without opening Test for model evaluation."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

from src.splits import load_raw_records, load_frozen_splits, build_repost_group_key
from src.approach_1_linear_sgd.pipeline import LinearSGDPipeline
from src.approach_1_linear_sgd.trainer import score


def verify_reload(run_path):
    """Worker: reproduce validation predictions using only the saved inference pipeline."""
    run = json.loads(run_path.read_text(encoding="utf-8"))
    root = run_path.parent.parent
    saved_path = run["artifacts"].get("pipeline_path")
    if not saved_path:
        raise ValueError("This legacy run has no saved pipeline; train a new run to verify reload")
    pipeline = LinearSGDPipeline.load(root / saved_path)
    rows = load_raw_records()
    _, vi, _ = load_frozen_splits(rows=rows)
    val_rows = [rows[int(i)] for i in vi]
    prices = np.array([float(r["Price"]) for r in val_rows])
    # Never fit anything here; pipeline removes the labels before transform.
    actual = score(prices, pipeline.predict(val_rows))
    expected = run["val_metrics"]
    for name in ("rmse", "mae", "r2", "within_10pct", "within_20pct"):
        if abs(actual[name] - expected[name]) >= 1e-5:
            raise ValueError(f"Reload changed validation metric {name}: {actual[name]} vs {expected[name]}")
    checkpoint = torch.load(root / "checkpoints" / f"{run['run_id']}_best.pt",
                            map_location="cpu", weights_only=True)
    for key, tensor in pipeline.regressor.model.state_dict().items():
        torch.testing.assert_close(tensor, checkpoint["state_dict"][key], rtol=0, atol=0)
    if pipeline.regressor.scaler.n_samples_seen_ != run["splits"]["n_train"]:
        raise ValueError("Restored scaler sample count does not match Train")
    print(json.dumps(dict(run_id=run["run_id"], val_rmse=actual["rmse"], reload="passed")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", action="append", help="Verify specified runs instead of the original six")
    parser.add_argument("--reload-worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.reload_worker:
        verify_reload(args.reload_worker)
        return
    root = PROJECT_ROOT / "artifacts" / "approach_1_linear_sgd"
    rows = load_raw_records()
    indices = load_frozen_splits(rows=rows)
    groups = [{build_repost_group_key(rows[int(i)]) for i in part} for part in indices]
    for a, b in ((0, 1), (0, 2), (1, 2)):
        assert groups[a].isdisjoint(groups[b])
    names = args.run_id or ("L01", "L02", "L03", "L04a", "L04b", "L05")
    for name in names:
        if Path(name).name != name or name in (".", ".."):
            parser.error("Invalid run ID")
        run = json.loads((root / "runs" / f"{name}.json").read_text(encoding="utf-8"),
                         parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        assert run["metrics"]["test"] is None and run["eval_test"] is False
        assert run["scaler_n_samples_seen"] == len(indices[0])
        assert run["input_dim"] == 345
        assert run["cpu_config"] == dict(num_threads=4, num_workers=0, dtype="float32")
        with Path(run["artifacts"]["curve_path"]).open(encoding="utf-8", newline="") as fh:
            history = list(csv.DictReader(fh))
        assert len(history) == run["epochs_ran"]
        if run["best_epoch"]:
            checkpoint = torch.load(run["artifacts"]["checkpoint_path"], map_location="cpu", weights_only=True)
            assert checkpoint["best_epoch"] == run["best_epoch"]
            assert checkpoint["input_dim"] == run["input_dim"]
            assert abs(checkpoint["best_val_rmse"] - run["val_metrics"]["rmse"]) < 1e-5
            best_row = history[run["best_epoch"] - 1]
            assert abs(float(best_row["val_rmse"]) - run["best_val_rmse"]) < 1e-5
            if run["artifacts"].get("pipeline_path"):
                child = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                        "--reload-worker", str(root / "runs" / f"{name}.json")],
                                       capture_output=True, text=True, check=True,
                                       cwd=PROJECT_ROOT, timeout=120)
                print(child.stdout.strip())
            else:
                print(f"{name}: legacy checkpoint only; full pipeline reload not available")
        else:
            assert run["status"] == "diverged" and run["artifacts"]["checkpoint_path"] is None
        print(f"{name}: {run['status']}; {len(history)} epochs; Test locked; checks passed")
    text = (root / "leaderboard.md").read_text(encoding="utf-8")
    assert all(name in text for name in names)
    print(f"Verified {len(names)} JSON logs, histories, checkpoints, scaler counts and disjoint repost groups.")


if __name__ == "__main__":
    main()
