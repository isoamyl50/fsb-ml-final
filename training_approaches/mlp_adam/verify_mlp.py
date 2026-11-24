"""Run this CLI in a fresh Python process to verify the saved validation score."""

import argparse
import json
from pathlib import Path

try:
    from .train import (HERE, Config, configure_cpu, inverse_target, predict_batches,
                        read_training_csv, score, sha256, tf)
except ImportError:
    from train import (HERE, Config, configure_cpu, inverse_target, predict_batches,
                       read_training_csv, score, sha256, tf)


def verify(run_id, output_root=HERE):
    Config(run_id=run_id).validate()
    run = json.loads((Path(output_root) / "results" / "runs" / f"{run_id}.json").read_text(encoding="utf-8"))
    cpu = run["cpu_config"]
    configure_cpu(Config(seed=run["seed"], intra_threads=cpu["intra_op_threads"], inter_threads=cpu["inter_op_threads"]))
    validation = Path(run["data"]["normalized_dir"]) / "validation.csv"
    if sha256(validation) != run["data"]["val_csv_sha256"]:
        raise ValueError("Validation CSV hash has changed")
    names, x, y = read_training_csv(validation)
    if names != run["data"]["feature_names"] or len(x) != run["data"]["n_val"]:
        raise ValueError("Validation schema/row count has changed")
    model = tf.keras.models.load_model(run["artifacts"]["model_path"], compile=False)
    predictions = inverse_target(predict_batches(model, x), run["hyperparameters"]["target_transform"])
    actual = score(y, predictions)
    difference = abs(actual["rmse"] - run["val_metrics"]["rmse"])
    if difference >= 1e-5:
        raise ValueError(f"Reload RMSE mismatch: {difference}")
    print(json.dumps({"run_id": run_id, "verified": True, "val_rmse": actual["rmse"], "rmse_abs_diff": difference}))
    return actual


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=HERE)
    args = parser.parse_args()
    verify(args.run_id, args.output_root)
