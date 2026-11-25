"""Evaluate the saved three-hidden-layer MLP on the held-out Test CSV."""

import csv

import tensorflow as tf

from data_normalize import DataNormalizer, NORMALIZED_DIR, read_split
from data_split import SPLIT_DIR, TARGET
from train import CPU_THREADS, MODEL_PATH, configure_cpu, print_scores


def evaluate_test() -> None:
    configure_cpu()
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

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

    print(f"Saved model: {MODEL_PATH.name}")
    print(f"Test listings: {len(test_rows):,} | CPU: {CPU_THREADS} compute threads")
    print_scores("Test", test_targets, predictions)


if __name__ == "__main__":
    evaluate_test()
