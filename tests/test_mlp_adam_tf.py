"""Behavioral tests for the standalone TF benchmark; baseline needs no TensorFlow."""

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from training_approaches.mlp_adam import train as mlp

TF_AVAILABLE = importlib.util.find_spec("tensorflow") is not None


def synthetic_csvs(folder):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for partition, count in (("train", 24), ("validation", 8)):
        x = rng.normal(size=(count, 3)).astype(np.float32)
        y = 4.0 + 0.3 * x[:, 0] + 0.2 * np.maximum(x[:, 1], 0)
        with (folder / f"{partition}.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Area", "Floors", "Bedrooms", "Price"])
            writer.writerows(np.column_stack([x, y]))


class DataContractTests(unittest.TestCase):
    def test_copied_preprocessing_is_byte_identical(self):
        for name in ("data_split.py", "data_normalize.py"):
            self.assertEqual((mlp.HERE / name).read_bytes(), (mlp.HERE.parent / "linear_sgd" / name).read_bytes())

    def test_csv_shape_finite_schema_and_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            synthetic_csvs(folder)
            arrays, metadata = mlp.load_data(folder, folder)
            x, y = arrays["train"]
            self.assertEqual(x.shape, (24, 3))
            self.assertEqual(y.shape, (24, 1))
            self.assertEqual(x.dtype, np.float32)
            self.assertEqual(y.dtype, np.float32)
            self.assertEqual(metadata["input_dim"], 3)
            self.assertNotIn("Price", metadata["feature_names"])
            other = folder / "other"
            synthetic_csvs(other)
            with (other / "train.csv").open("a") as stream:
                stream.write("1,2,3,4\n")
            with self.assertRaisesRegex(ValueError, "parity"):
                mlp.load_data(folder, other)
            for text in ("Area,Price\nNaN,1\n2,3\n", "Area,Price\n1e100,1\n2,3\n",
                         "Price,Area\n1,2\n3,4\n", "Price,Price\n1,2\n3,4\n"):
                (folder / "bad.csv").write_text(text)
                with self.assertRaises(ValueError):
                    mlp.read_training_csv(folder / "bad.csv")
            (folder / "validation.csv").write_text("A,B,C,Price\n1,2,3,4\n2,3,4,5\n")
            with self.assertRaisesRegex(ValueError, "names/order"):
                mlp.load_data(folder, folder)

    def test_target_roundtrip_and_metric_units(self):
        y = np.array([[2.0], [4.0], [8.0]], dtype=np.float32)
        transformed = mlp.transform_target(y, "log1p")
        np.testing.assert_allclose(mlp.inverse_target(transformed, "log1p"), y.ravel(), atol=1e-6)
        expected = np.sqrt(np.mean((y.ravel() - np.array([3, 5, 7])) ** 2))
        predictions = mlp.inverse_target(np.log1p([3, 5, 7]), "log1p")
        self.assertAlmostEqual(mlp.score(y, predictions)["rmse"], expected)
        with self.assertRaises(ValueError):
            mlp.transform_target(-y, "log1p")
        with self.assertRaises(ValueError):
            mlp.inverse_target([np.inf], "log1p")

    def test_cli_override_and_config_cannot_unlock_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"run_id": "original", "seed": 42, "hidden_units": 32}))
            args, config = mlp.parse_args(["--config", str(path), "--seed", "123", "--run-id", "new"])
            self.assertEqual((config.seed, config.run_id, config.hidden_units), (123, "new", 32))
            self.assertFalse(args.eval_test)
            path.write_text('{"eval_test": true}')
            with self.assertRaises(SystemExit):
                mlp.parse_args(["--config", str(path)])
            with self.assertRaises(ValueError):
                mlp.Config(run_id="../escape").validate()


@unittest.skipUnless(TF_AVAILABLE, "TensorFlow is not installed")
class MlpTensorFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mlp.configure_cpu(mlp.Config(intra_threads=2))

    def config(self, run_id, **kwargs):
        return mlp.Config(run_id=run_id, max_epochs=2, patience=2, hidden_units=4,
                          batch_size=24, intra_threads=2, **kwargs)

    def test_architecture_initialization_and_shape(self):
        tf = mlp.tf
        for dropout in (0.0, 0.1):
            model = mlp.build_mlp_model(3, 8, dropout, 42)
            self.assertEqual(len(model.layers), 3 if dropout else 2)
            hidden, output = model.layers[0], model.layers[-1]
            self.assertIsInstance(hidden.kernel_initializer, tf.keras.initializers.HeNormal)
            self.assertEqual(hidden.activation.__name__, "relu")
            self.assertTrue(np.any(hidden.kernel.numpy() != 0))
            self.assertEqual(output.activation.__name__, "linear")
            result = model(np.zeros((5, 3), np.float32))
            self.assertEqual(tuple(result.shape), (5, 1))
            self.assertEqual(result.dtype, tf.float32)

    def test_checkpoint_reproducibility_test_lock_and_fresh_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            data = folder / "normalized"
            synthetic_csvs(data)
            reports = [mlp.train(self.config(name, target_transform="log1p"), data, data, folder, verbose=0)
                       for name in ("first", "repeat")]
            self.assertIsNone(reports[0]["test_metrics"])
            self.assertTrue(reports[0]["reload_check"]["passed"])
            self.assertAlmostEqual(reports[0]["val_metrics"]["rmse"], reports[1]["val_metrics"]["rmse"], places=7)
            _, xv, yv = mlp.read_training_csv(data / "validation.csv")
            models = [mlp.tf.keras.models.load_model(r["artifacts"]["model_path"], compile=False) for r in reports]
            predictions = [mlp.predict_batches(model, xv) for model in models]
            np.testing.assert_allclose(*predictions, atol=1e-7)
            price_rmse = mlp.score(yv, mlp.inverse_target(predictions[0], "log1p"))["rmse"]
            self.assertAlmostEqual(price_rmse, reports[0]["val_metrics"]["rmse"], places=7)
            self.assertGreater(abs(price_rmse - mlp.score(np.log1p(yv), predictions[0])["rmse"]), 0.1)
            with self.assertRaises(FileExistsError):
                mlp.train(self.config("first"), data, data, folder, verbose=0)
            process = subprocess.run([sys.executable, str(mlp.HERE / "verify_mlp.py"), "--run-id", "first",
                                      "--output-root", str(folder)], capture_output=True, text=True, timeout=90)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertIn('"verified": true', process.stdout)
            self.assertFalse((data / "test.csv").exists())
            (data / "validation.csv").write_text("changed")
            from training_approaches.mlp_adam.verify_mlp import verify
            with self.assertRaisesRegex(ValueError, "hash"):
                verify("first", folder)

    def test_divergence_preserves_last_finite_checkpoint_and_returns_nonzero(self):
        original = mlp.VndMonitorAndCheckpointCallback.on_train_batch_end
        retained = {}

        def inject_nonfinite(callback, batch, logs=None):
            if callback.epochs_attempted == 2:
                retained["hash"] = mlp.sha256(callback.model_path)
                callback.model.layers[0].kernel.assign(np.full(callback.model.layers[0].kernel.shape, np.inf, np.float32))
                logs = {"loss": float("inf")}
            original(callback, batch, logs)

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            data = folder / "normalized"
            synthetic_csvs(data)
            with patch.object(mlp.VndMonitorAndCheckpointCallback, "on_train_batch_end", inject_nonfinite):
                report = mlp.train(self.config("diverged", lr=1e6), data, data, folder, verbose=0)
            self.assertEqual(report["status"], "diverged")
            self.assertEqual(report["diverged_epoch"], 2)
            self.assertTrue(report["reload_check"]["passed"])
            self.assertEqual(mlp.sha256(report["artifacts"]["model_path"]), retained["hash"])
            self.assertIsNone(report["test_metrics"])
            with patch.object(mlp, "train", return_value=report):
                self.assertEqual(mlp.main(["--quiet"]), 1)

    def test_missing_loss_or_nonfinite_prediction_stops_before_saving(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = (np.ones((2, 3), np.float32), np.array([[2], [3]], np.float32))
            monitor = mlp.VndMonitorAndCheckpointCallback(data, data, self.config("guard"), root / "best.keras", root / "history.csv")
            monitor.set_model(mlp.build_mlp_model(3))
            monitor.on_epoch_begin(0)
            monitor.on_train_batch_end(0, {})
            self.assertEqual(monitor.status, "diverged")
            self.assertFalse(monitor.model_path.exists())
            monitor.status = "completed"
            with patch.object(mlp, "predict_batches", return_value=np.array([np.nan, 1])):
                monitor.on_epoch_end(0, {"loss": 1.0})
            self.assertEqual(monitor.status, "diverged")
            self.assertFalse(monitor.model_path.exists())

    def test_early_stopping_and_interrupted_history_are_durable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = (np.ones((2, 3), np.float32), np.array([[2], [3]], np.float32))
            monitor = mlp.VndMonitorAndCheckpointCallback(data, data, self.config("stop"), root / "best.keras", root / "history.csv")
            monitor.set_model(mlp.build_mlp_model(3, 4))
            monitor.on_train_begin()
            try:
                for epoch in range(3):
                    monitor.on_epoch_begin(epoch)
                    monitor.on_epoch_end(epoch, {"loss": 1.0})
                self.assertEqual(monitor.status, "early_stopped")
                self.assertEqual(monitor.best_epoch, 1)
                with (root / "history.csv").open() as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 3)
            finally:
                monitor.close()

    def test_final_test_uses_train_normalizer_and_can_only_run_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split = root / "dataset" / "split"
            normalized = split.parent / "normalized"
            split.mkdir(parents=True)
            normalized.mkdir()
            (root / "results").mkdir()
            module = mlp.load_normalizer_module()
            columns = list(module.FEATURE_COLUMNS) + ["Price"]
            for name, areas in (("train", [10, 20, 30]), ("validation", [15, 25]), ("test", [100, 200])):
                with (split / f"{name}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=columns)
                    writer.writeheader()
                    for area in areas:
                        row = {column: "" for column in columns}
                        row.update(Address="Street, District, City", Area=str(area), Frontage="3",
                                   Floors="2", Bedrooms="2", Bathrooms="1", Price="4")
                        row["Access Road"] = "5"
                        writer.writerow(row)
            train_rows, _ = module.read_split(split / "train.csv")
            normalizer = module.DataNormalizer(train_rows)
            for name in ("train", "validation"):
                rows, prices = module.read_split(split / f"{name}.csv")
                with (normalized / f"{name}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(normalizer.feature_names + ["Price"])
                    writer.writerows([features + [price] for features, price in zip(normalizer.transform(rows), prices)])
            _, data = mlp.load_data(normalized, normalized)
            model = mlp.build_mlp_model(data["input_dim"], 4)
            original = module.DataNormalizer.transform
            seen_test = []

            def observe(instance, rows):
                if rows[0]["Area"] == "100":
                    seen_test.append(instance.means["Area"])
                return original(instance, rows)

            with patch.object(mlp, "load_normalizer_module", return_value=module), patch.object(module.DataNormalizer, "transform", observe):
                metrics = mlp.evaluate_locked_test(model, self.config("final"), data, root)
                self.assertTrue(np.isfinite(metrics["rmse"]))
                self.assertEqual(seen_test, [20.0])
                with self.assertRaises(FileExistsError):
                    mlp.evaluate_locked_test(model, self.config("second"), data, root)
                self.assertEqual(seen_test, [20.0])


if __name__ == "__main__":
    unittest.main()
