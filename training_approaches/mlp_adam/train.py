"""Standalone CPU MLP + Adam; reuse Linear SGD's normalized matrices unchanged."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

# Set before importing TensorFlow. No GPU is used in this benchmark.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np

try:
    import tensorflow as tf
except ModuleNotFoundError as exc:
    if exc.name != "tensorflow":
        raise
    tf = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REFERENCE_DIR = HERE.parent / "linear_sgd" / "dataset" / "normalized"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate import evaluate_predictions
from src.experiment_io import atomic_write_text, file_lock


@dataclass
class Config:
    run_id: str = "N01"
    seed: int = 42
    hidden_units: int = 64
    batch_size: int = 256
    lr: float = 0.001
    dropout: float = 0.0
    target_transform: str = "raw"
    loss: str = "mse"
    huber_delta: float = 1.0
    max_epochs: int = 150
    patience: int = 15
    intra_threads: int = 6
    inter_threads: int = 1

    def validate(self):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.run_id):
            raise ValueError("run_id must be a simple filename, without path separators")
        for key in ("hidden_units", "batch_size", "max_epochs", "patience",
                    "intra_threads", "inter_threads"):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be finite and positive")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.target_transform not in ("raw", "log1p"):
            raise ValueError("target_transform must be raw or log1p")
        if self.loss not in ("mse", "huber"):
            raise ValueError("loss must be mse or huber")
        if not math.isfinite(self.huber_delta) or self.huber_delta <= 0:
            raise ValueError("huber_delta must be finite and positive")


def require_tensorflow():
    if tf is None:
        raise RuntimeError("TensorFlow is not installed. Install this approach's requirements.txt")


def configure_cpu(config):
    require_tensorflow()
    tf.config.set_visible_devices([], "GPU")
    for getter, setter, value in (
        (tf.config.threading.get_intra_op_parallelism_threads,
         tf.config.threading.set_intra_op_parallelism_threads, config.intra_threads),
        (tf.config.threading.get_inter_op_parallelism_threads,
         tf.config.threading.set_inter_op_parallelism_threads, config.inter_threads),
    ):
        if getter() != value:
            setter(value)  # Fail clearly if the caller has already initialized TF.
    os.environ["PYTHONHASHSEED"] = str(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    tf.keras.utils.set_random_seed(config.seed)
    tf.config.experimental.enable_op_determinism()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_training_csv(path):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        columns = next(reader, [])
        if len(columns) < 2 or columns[-1] != "Price":
            raise ValueError(f"{path} must end with Price")
        if len(set(columns)) != len(columns):
            raise ValueError(f"Duplicate column names in {path}")
        rows = []
        for row in reader:
            if len(row) != len(columns):
                raise ValueError(f"Wrong number of fields at {path}:{reader.line_num}")
            rows.append([float(value) for value in row])
    if len(rows) < 2:
        raise ValueError(f"{path} needs at least two rows for regression metrics")
    with np.errstate(over="ignore"):
        matrix = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{path} contains NaN or Inf (including float32 overflow)")
    return columns[:-1], matrix[:, :-1].copy(), matrix[:, -1:].copy()


def load_data(normalized_dir, reference_dir):
    """Read only Train/Validation; require byte-for-byte parity with the reference."""
    normalized_dir, reference_dir = Path(normalized_dir).resolve(), Path(reference_dir).resolve()
    arrays, metadata = {}, {"normalized_dir": str(normalized_dir),
                            "reference_dir": str(reference_dir), "parity_verified": True}
    feature_names = None
    for partition, short in (("train", "train"), ("validation", "val")):
        path = normalized_dir / f"{partition}.csv"
        digest = sha256(path)
        reference_hash = sha256(reference_dir / path.name)
        if digest != reference_hash:
            raise ValueError(f"Data parity failed: {path.name} differs from Linear SGD reference")
        columns, x, y = read_training_csv(path)
        if feature_names is not None and columns != feature_names:
            raise ValueError("Train and Validation feature names/order differ")
        feature_names = columns
        arrays[short] = (x, y)
        metadata.update({f"{short}_csv_sha256": digest, f"n_{short}": len(x),
                         f"{short}_n_columns": len(columns) + 1})
    metadata.update(input_dim=len(feature_names), feature_names=feature_names)
    return arrays, metadata


def transform_target(y, name):
    if name == "log1p":
        if np.any(y < 0):
            raise ValueError("log1p target requires nonnegative prices")
        return np.log1p(y).astype(np.float32)
    if name != "raw":
        raise ValueError(f"Unknown target transform: {name}")
    return np.asarray(y, dtype=np.float32).reshape(-1, 1)


def inverse_target(predictions, name):
    values = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("Model predictions contain NaN or Inf")
    if name == "log1p":
        return np.expm1(np.clip(values, -20.0, 20.0))
    if name != "raw":
        raise ValueError(f"Unknown target transform: {name}")
    return values


def score(y, predictions):
    return evaluate_predictions("mlp_adam_tf", np.asarray(y).reshape(-1),
                                np.asarray(predictions).reshape(-1), full_precision=True)


def build_mlp_model(input_dim, hidden_units=64, dropout_rate=0.0, seed=42):
    require_tensorflow()
    layers = [tf.keras.layers.Input(shape=(input_dim,), dtype="float32", name="features"),
              tf.keras.layers.Dense(hidden_units, activation="relu", dtype="float32",
                                    kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
                                    bias_initializer="zeros", name="hidden_dense")]
    if dropout_rate > 0:
        layers.append(tf.keras.layers.Dropout(dropout_rate, seed=seed, name="hidden_dropout"))
    layers.append(tf.keras.layers.Dense(1, activation="linear", dtype="float32",
                                       kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                                       bias_initializer="zeros", name="price_output"))
    return tf.keras.Sequential(layers, name="mlp_adam")


def predict_batches(model, features, batch_size=1024):
    return np.concatenate([np.asarray(model(features[i:i + batch_size], training=False)).reshape(-1)
                           for i in range(0, len(features), batch_size)])


class VndMonitorAndCheckpointCallback(tf.keras.callbacks.Callback if tf else object):
    """Use original-price metrics for selection, including for log-target training."""

    FIELDS = ["epoch", "train_loss", "train_rmse", "train_mae", "train_r2",
              "train_within_10pct", "train_within_20pct", "val_rmse", "val_mae",
              "val_r2", "val_within_10pct", "val_within_20pct", "train_val_gap", "epoch_sec"]

    def __init__(self, train, validation, config, model_path, curve_path):
        super().__init__()
        self.xt, self.yt = train
        self.xv, self.yv = validation
        self.config = config
        self.model_path, self.curve_path = Path(model_path), Path(curve_path)
        self.status = "completed"
        self.diverged_epoch = None
        self.best_val_rmse = math.inf
        self.best_epoch = self.no_improve = self.epochs_attempted = 0
        self.best_train_metrics = self.best_val_metrics = None
        self.history = []
        self.stream = None

    def on_train_begin(self, logs=None):
        self.curve_path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.curve_path.open("x", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.stream, fieldnames=self.FIELDS)
        self.writer.writeheader()
        self.flush()

    def flush(self):
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def on_train_end(self, logs=None):
        self.close()

    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.perf_counter()
        self.epochs_attempted = epoch + 1

    def mark_diverged(self):
        self.status = "diverged"
        self.diverged_epoch = self.epochs_attempted
        self.model.stop_training = True

    def on_train_batch_end(self, batch, logs=None):
        loss = (logs or {}).get("loss")
        if loss is None or not np.isfinite(loss):
            self.mark_diverged()

    def on_epoch_end(self, epoch, logs=None):
        if self.status == "diverged":
            return
        loss = (logs or {}).get("loss")
        if loss is None or not np.isfinite(loss):
            self.mark_diverged()
            return
        pt = predict_batches(self.model, self.xt)
        pv = predict_batches(self.model, self.xv)
        if not np.isfinite(pt).all() or not np.isfinite(pv).all():
            self.mark_diverged()
            return
        mt = score(self.yt, inverse_target(pt, self.config.target_transform))
        mv = score(self.yv, inverse_target(pv, self.config.target_transform))
        if not all(math.isfinite(m[k]) for m in (mt, mv) for k in ("rmse", "mae", "r2")):
            self.mark_diverged()
            return
        improved = mv["rmse"] < self.best_val_rmse - 1e-6
        if improved:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.model_path.with_suffix(".tmp.keras")
            try:
                self.model.save(temporary)
                os.replace(temporary, self.model_path)
            finally:
                temporary.unlink(missing_ok=True)
            self.best_val_rmse, self.best_epoch = mv["rmse"], epoch + 1
            self.best_train_metrics, self.best_val_metrics = mt, mv
            self.no_improve = 0
        else:
            self.no_improve += 1
            if self.no_improve >= self.config.patience:
                self.status = "early_stopped"
                self.model.stop_training = True
        row = {"epoch": epoch + 1, "train_loss": float(loss),
               "train_val_gap": mv["rmse"] - mt["rmse"], "epoch_sec": time.perf_counter() - self.t0}
        for prefix, metrics in (("train", mt), ("val", mv)):
            row.update({f"{prefix}_{key}": metrics[key] for key in
                        ("rmse", "mae", "r2", "within_10pct", "within_20pct")})
        self.writer.writerow(row)
        self.flush()
        self.history.append(row)
        if logs is not None:
            logs.update(val_rmse=mv["rmse"], train_rmse=mt["rmse"])


def write_json(path, document):
    atomic_write_text(path, json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def build_leaderboard(output_root):
    results = Path(output_root) / "results"
    with file_lock(results / ".leaderboard.lock", timeout=30):
        runs = [json.loads(path.read_text(encoding="utf-8")) for path in (results / "runs").glob("*.json")]
        runs.sort(key=lambda run: (run["val_metrics"] or {}).get("rmse", math.inf))
        lines = ["# MLP + Adam — TensorFlow CPU", "",
                 "Cùng split/normalize với `training_approaches/linear_sgd` TensorFlow; "
                 "đối chiếu Linear SGD TF bằng cùng dữ liệu và metric Validation. "
                 "Không thuộc benchmark P4 trong `experiments/leaderboard.md`.", "",
                 "| Run ID | Hidden | Batch | LR | Dropout | Target | Seed | Best Epoch | Val RMSE (tỷ) | Val MAE (tỷ) | Val R2 | Within 10% | Train-Val Gap | Sec/Epoch | Total Sec | Status |",
                 "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for run in runs:
            hp, timing, metrics = run["hyperparameters"], run["timing"], run["val_metrics"] or {}
            values = [run["run_id"], hp["hidden_units"], hp["batch_size"], hp["lr"], hp["dropout"],
                      hp["target_transform"], run["seed"], timing["best_epoch"], metrics.get("rmse"),
                      metrics.get("mae"), metrics.get("r2"), metrics.get("within_10pct"),
                      run["train_val_gap"], timing["sec_per_epoch"], timing["total_train_sec"], run["status"]]
            lines.append("| " + " | ".join("—" if v is None else f"{v:.6f}" if isinstance(v, float)
                                          else str(v) for v in values) + " |")
        lines.extend(["", "Độ ổn định đa seed (độ lệch chuẩn mẫu, ddof=1):", ""])
        groups = {}
        for run in runs:
            if run["status"] not in ("completed", "early_stopped") or not run["val_metrics"]:
                continue
            hp = dict(run["hyperparameters"])
            hp.pop("seed", None)
            hp.pop("run_id", None)
            key = (json.dumps(hp, sort_keys=True), run["data"]["train_csv_sha256"], run["data"]["val_csv_sha256"])
            # Repeats of the same seed do not count as independent observations.
            groups.setdefault(key, {})[run["seed"]] = run
        for seeds in groups.values():
            if {42, 123, 2026}.issubset(seeds):
                selected = [seeds[s] for s in (42, 123, 2026)]
                details = []
                for metric in ("rmse", "mae", "r2", "within_10pct"):
                    values = [r["val_metrics"][metric] for r in selected]
                    details.append(f"{metric}: {np.mean(values):.6f} ± {np.std(values, ddof=1):.6f}")
                lines.append("- " + ", ".join(r["run_id"] for r in selected) + ": " + "; ".join(details))
        atomic_write_text(results / "leaderboard.md", "\n".join(lines) + "\n")


def load_normalizer_module():
    """Load the unchanged script without leaving its bare import in sys.modules."""
    # The copied normalizer uses the original script-style import of data_split.
    import importlib.util
    split_spec = importlib.util.spec_from_file_location("data_split", HERE / "data_split.py")
    split_module = importlib.util.module_from_spec(split_spec)
    previous = sys.modules.get("data_split")
    try:
        split_spec.loader.exec_module(split_module)
        sys.modules["data_split"] = split_module
        normalizer_spec = importlib.util.spec_from_file_location("mlp_data_normalize", HERE / "data_normalize.py")
        normalizer_module = importlib.util.module_from_spec(normalizer_spec)
        normalizer_spec.loader.exec_module(normalizer_module)
    finally:
        if previous is None:
            sys.modules.pop("data_split", None)
        else:
            sys.modules["data_split"] = previous
    return normalizer_module


def evaluate_locked_test(model, config, data, output_root):
    """Reconstruct preprocessing from Train only; reserve a single final Test read."""
    normalizer_module = load_normalizer_module()
    normalized_dir = Path(data["normalized_dir"])
    split_dir = normalized_dir.parent / "split"
    train_rows, _ = normalizer_module.read_split(split_dir / "train.csv")
    normalizer = normalizer_module.DataNormalizer(train_rows)
    if normalizer.feature_names != data["feature_names"]:
        raise ValueError("Reconstructed normalizer has different feature columns")
    # Establish that raw Train really reconstructs the matrices used for training.
    for partition in ("train", "validation"):
        rows, prices = normalizer_module.read_split(split_dir / f"{partition}.csv")
        _, x, y = read_training_csv(normalized_dir / f"{partition}.csv")
        np.testing.assert_array_equal(np.asarray(normalizer.transform(rows), dtype=np.float32), x)
        np.testing.assert_array_equal(np.asarray(prices, dtype=np.float32), y.ravel())
    reservation = Path(output_root) / "results" / "test_evaluation.json"
    # Persistent reservation: even a failed/interrupted final evaluation is never silently retried.
    with reservation.open("x", encoding="utf-8") as stream:
        json.dump({"run_id": config.run_id, "status": "reserved",
                   "train_csv_sha256": data["train_csv_sha256"]}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    rows, y = normalizer_module.read_split(split_dir / "test.csv")
    x = np.asarray(normalizer.transform(rows), dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Test contains nonfinite features")
    metrics = score(y, inverse_target(predict_batches(model, x), config.target_transform))
    write_json(reservation, {"run_id": config.run_id, "status": "evaluated", "test_metrics": metrics})
    return metrics


def train(config, normalized_dir=None, reference_dir=REFERENCE_DIR, output_root=HERE,
          eval_test=False, verbose=2):
    config.validate()
    require_tensorflow()
    output_root = Path(output_root).resolve()
    local = HERE / "dataset" / "normalized"
    normalized_dir = Path(normalized_dir) if normalized_dir else (local if (local / "train.csv").exists() else REFERENCE_DIR)
    results = output_root / "results"
    run_path = results / "runs" / f"{config.run_id}.json"
    model_path = output_root / "output_model" / f"{config.run_id}_best.keras"
    curve_path = results / "curves" / f"{config.run_id}_history.csv"
    with file_lock(results / ".locks" / f"{config.run_id}.lock"):
        if any(path.exists() for path in (run_path, model_path, curve_path)):
            raise FileExistsError(f"Run {config.run_id} already exists; choose a new --run-id")
        if eval_test and (results / "test_evaluation.json").exists():
            raise FileExistsError("Final Test evaluation is already reserved/evaluated")
        arrays, data = load_data(normalized_dir, reference_dir)
        xt, yt = arrays["train"]
        xv, yv = arrays["val"]
        y_loss = transform_target(yt, config.target_transform)
        transform_target(yv, config.target_transform)  # Validate the target contract without fitting.
        configure_cpu(config)
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(config.seed)
        model = build_mlp_model(data["input_dim"], config.hidden_units, config.dropout, config.seed)
        loss = tf.keras.losses.MeanSquaredError() if config.loss == "mse" else tf.keras.losses.Huber(delta=config.huber_delta)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.lr), loss=loss)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        options.threading.max_intra_op_parallelism = config.intra_threads
        dataset = (tf.data.Dataset.from_tensor_slices((xt, y_loss))
                   .shuffle(len(xt), seed=config.seed, reshuffle_each_iteration=True)
                   .batch(config.batch_size).with_options(options).prefetch(1))
        monitor = VndMonitorAndCheckpointCallback((xt, yt), (xv, yv), config, model_path, curve_path)
        started = time.perf_counter()
        try:
            model.fit(dataset, epochs=config.max_epochs, callbacks=[monitor], shuffle=False, verbose=verbose)
        except KeyboardInterrupt:
            monitor.status = "interrupted"
        finally:
            monitor.close()
        elapsed = time.perf_counter() - started
        reload_check = None
        best_model = None
        if model_path.exists():
            best_model = tf.keras.models.load_model(model_path, compile=False)
            metrics = score(yv, inverse_target(predict_batches(best_model, xv), config.target_transform))
            difference = abs(metrics["rmse"] - monitor.best_val_rmse)
            if difference >= 1e-5:
                raise RuntimeError(f"Checkpoint reload RMSE mismatch: {difference}")
            reload_check = {"passed": True, "rmse_abs_diff": difference}
        mt, mv = monitor.best_train_metrics, monitor.best_val_metrics
        report = {"run_id": config.run_id, "approach": "training_approaches_mlp_adam_tf",
                  "benchmark_scope": "linear_sgd_tf_same_split_and_normalization_only",
                  "timestamp_utc": datetime.now(timezone.utc).isoformat(), "seed": config.seed,
                  "status": monitor.status, "diverged_epoch": monitor.diverged_epoch,
                  "data": data, "cpu_config": {"intra_op_threads": config.intra_threads,
                  "inter_op_threads": config.inter_threads, "dtype": "float32"},
                  "versions": {"tensorflow": tf.__version__, "numpy": np.__version__, "python": sys.version},
                  "hyperparameters": {**asdict(config), "activation": "relu", "optimizer": "adam"},
                  "timing": {"epochs_run": monitor.epochs_attempted, "epochs_logged": len(monitor.history),
                             "best_epoch": monitor.best_epoch, "total_train_sec": elapsed,
                             "sec_per_epoch": float(np.mean([r["epoch_sec"] for r in monitor.history])) if monitor.history else 0.0},
                  "train_metrics": mt, "val_metrics": mv, "test_metrics": None,
                  "train_val_gap": mv["rmse"] - mt["rmse"] if mv else None,
                  "reload_check": reload_check,
                  "artifacts": {"model_path": str(model_path) if best_model else None, "curve_path": str(curve_path)}}
        # Persist tuning results before any explicit final Test action.
        write_json(run_path, report)
        if eval_test and best_model is not None and monitor.status in ("completed", "early_stopped"):
            with file_lock(results / ".test.lock"):
                report["test_metrics"] = evaluate_locked_test(best_model, config, data, output_root)
                write_json(run_path, report)
        build_leaderboard(output_root)
        return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON/YAML; CLI arguments override config")
    for name, field in Config.__dataclass_fields__.items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(field.default), default=None)
    parser.add_argument("--normalized-dir", type=Path)
    parser.add_argument("--reference-dir", type=Path, default=REFERENCE_DIR,
                        help="Unchanged Linear SGD normalized CSVs used for SHA256 parity")
    parser.add_argument("--output-root", type=Path, default=HERE)
    parser.add_argument("--eval-test", action="store_true", help="Final locked model only; single Test evaluation")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    values = {}
    if args.config:
        text = args.config.read_text(encoding="utf-8-sig")
        if args.config.suffix.lower() in (".yaml", ".yml"):
            import yaml
            values = yaml.safe_load(text)
        else:
            values = json.loads(text)
        if not isinstance(values, dict):
            parser.error("Config must be an object")
        unknown = set(values) - set(Config.__dataclass_fields__)
        if unknown:
            parser.error(f"Unknown config keys (Test can only be enabled by CLI): {sorted(unknown)}")
    values.update({name: getattr(args, name) for name in Config.__dataclass_fields__ if getattr(args, name) is not None})
    config = Config(**values)
    config.validate()
    return args, config


def main(argv=None):
    try:
        args, config = parse_args(argv)
        report = train(config, args.normalized_dir, args.reference_dir, args.output_root,
                       args.eval_test, verbose=0 if args.quiet else 2)
        print(json.dumps({"run_id": report["run_id"], "status": report["status"],
                          "val_rmse": (report["val_metrics"] or {}).get("rmse"),
                          "test_metrics": report["test_metrics"]}, ensure_ascii=False))
        return 130 if report["status"] == "interrupted" else 1 if report["status"] == "diverged" else 0
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
