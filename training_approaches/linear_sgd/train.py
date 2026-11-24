"""Train a linear house-price model with TensorFlow on CPU."""

import csv
import math
from pathlib import Path

import tensorflow as tf

from data_normalize import NORMALIZED_DIR
from data_split import TARGET


CPU_THREADS = 6
SEED = 42
BATCH_SIZE = 256
LEARNING_RATE = 0.05
MAX_EPOCHS = 200
PATIENCE = 20
MODEL_PATH = Path(__file__).resolve().parent / "output_model" / "linear_sgd_model.keras"


def read_training_csv(path: Path) -> tuple[list[str], tf.Tensor, tf.Tensor]:
    features: list[list[float]] = []
    prices: list[list[float]] = []
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        header = next(reader, [])
        column_count = len(header)
        price_column_index = column_count - 1
        if column_count < 2 or header[price_column_index] != TARGET:
            raise ValueError(f"{path.name} must end with a {TARGET} column.")

        feature_columns: list[str] = []
        for column_index in range(price_column_index):
            feature_columns.append(header[column_index])

        for row in reader:
            if len(row) != column_count:
                raise ValueError(f"Wrong number of fields in {path.name} at line {reader.line_num}.")

            feature_row: list[float] = []
            for column_index in range(price_column_index):
                feature_value = float(row[column_index])
                feature_row.append(feature_value)

            price = float(row[price_column_index])
            features.append(feature_row)
            prices.append([price])

    if not features:
        raise ValueError(f"{path.name} has no listings.")

    feature_tensor = tf.constant(features, dtype=tf.float32)
    price_tensor = tf.constant(prices, dtype=tf.float32)
    return feature_columns, feature_tensor, price_tensor


def train() -> None:
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(CPU_THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(SEED)
    print(f"TensorFlow {tf.__version__} | CPU | {CPU_THREADS} compute threads")

    train_path = NORMALIZED_DIR / "train.csv"
    validation_path = NORMALIZED_DIR / "validation.csv"
    train_columns, train_features, train_prices = read_training_csv(train_path)
    validation_columns, validation_features, validation_prices = read_training_csv(validation_path)
    if train_columns != validation_columns:
        raise ValueError("Normalized Train and Validation must have the same feature columns.")

    feature_count = len(train_columns)
    input_layer = tf.keras.layers.InputLayer(shape=(feature_count,))
    price_layer = tf.keras.layers.Dense(1, kernel_initializer="zeros")
    model = tf.keras.Sequential()
    model.add(input_layer)
    model.add(price_layer)

    optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
    loss_function = tf.keras.losses.MeanSquaredError()
    metrics = [
        tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        tf.keras.metrics.MeanAbsoluteError(name="mae"),
        tf.keras.metrics.R2Score(name="r2"),
    ]
    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_rmse",
        mode="min",
        patience=PATIENCE,
        restore_best_weights=True,
    )
    history = model.fit(
        train_features,
        train_prices,
        validation_data=(validation_features, validation_prices),
        batch_size=BATCH_SIZE,
        epochs=MAX_EPOCHS,
        callbacks=[early_stopping],
        verbose=2,
    )

    scores = model.evaluate(
        validation_features,
        validation_prices,
        batch_size=BATCH_SIZE,
        return_dict=True,
        verbose=0,
    )
    validation_rmse = scores["rmse"]
    if not math.isfinite(validation_rmse):
        raise RuntimeError("Validation RMSE is not finite.")

    rmse_by_epoch = history.history["val_rmse"]
    lowest_rmse = min(rmse_by_epoch)
    best_epoch = rmse_by_epoch.index(lowest_rmse) + 1

    predictions = model(validation_features, training=False)
    absolute_errors = tf.abs(predictions - validation_prices)
    allowed_errors = 0.10 * validation_prices
    within_ten_percent = absolute_errors <= allowed_errors
    within_ten_percent_as_numbers = tf.cast(within_ten_percent, tf.float32)
    accuracy = float(tf.reduce_mean(within_ten_percent_as_numbers))

    model.save(MODEL_PATH)
    print(f"Best epoch: {best_epoch}")
    print(f"Validation RMSE: {validation_rmse:.4f} billion VND")
    print(f"Validation MAE: {scores['mae']:.4f} billion VND")
    print(f"Validation R2: {scores['r2']:.4f}")
    print(f"Validation within 10%: {accuracy:.2%}")
    print(f"Saved: {MODEL_PATH.name}")


if __name__ == "__main__":
    train()
