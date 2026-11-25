"""Evaluate the saved Linear SGD model on the held-out Test CSV."""

import csv

import tensorflow as tf

from data_normalize import DataNormalizer, NORMALIZED_DIR, read_split
from data_split import SPLIT_DIR, TARGET
from train import CPU_THREADS, MODEL_PATH


def evaluate_test() -> None:
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(CPU_THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # Rebuild preprocessing from Train so Test cannot affect the scaling or encoding.
    train_rows, _ = read_split(SPLIT_DIR / "train.csv")
    normalizer = DataNormalizer(train_rows)
    expected_header = normalizer.feature_names.copy()
    expected_header.append(TARGET)
    with (NORMALIZED_DIR / "train.csv").open(encoding="utf-8-sig", newline="") as source:
        training_header = next(csv.reader(source), [])
    if training_header != expected_header:
        raise ValueError("Rebuilt feature columns do not match the normalized Train CSV.")
    if model.input_shape[-1] != len(normalizer.feature_names):
        raise ValueError("The saved model expects a different number of features.")

    test_rows, test_prices = read_split(SPLIT_DIR / "test.csv")
    if not test_rows:
        raise ValueError("Test has no listings.")

    test_matrix = normalizer.transform(test_rows)
    test_features = tf.constant(test_matrix, dtype=tf.float32)
    test_targets = tf.constant(test_prices, dtype=tf.float32)
    test_targets = tf.expand_dims(test_targets, axis=1)
    predictions = model(test_features, training=False)
    if predictions.shape != test_targets.shape:
        raise ValueError("The predictions must contain one price per Test listing.")
    tf.debugging.assert_all_finite(predictions, "Test predictions must be finite.")

    errors = predictions - test_targets
    rmse = float(tf.sqrt(tf.reduce_mean(tf.square(errors))))
    absolute_errors = tf.abs(errors)
    mae = float(tf.reduce_mean(absolute_errors))

    r2_metric = tf.keras.metrics.R2Score()
    r2_metric.update_state(test_targets, predictions)
    r2 = float(r2_metric.result())

    allowed_errors = 0.10 * test_targets
    within_ten_percent = absolute_errors <= allowed_errors
    within_ten_percent_as_numbers = tf.cast(within_ten_percent, tf.float32)
    accuracy = float(tf.reduce_mean(within_ten_percent_as_numbers))

    print(f"Saved model: {MODEL_PATH.name}")
    print(f"Test listings: {len(test_rows):,} | CPU: {CPU_THREADS} compute threads")
    print(f"Test RMSE: {rmse:.4f} billion VND")
    print(f"Test MAE: {mae:.4f} billion VND")
    print(f"Test R2: {r2:.4f}")
    print(f"Test within 10%: {accuracy:.2%}")


if __name__ == "__main__":
    evaluate_test()
