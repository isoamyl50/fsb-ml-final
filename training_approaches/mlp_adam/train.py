"""Train a house-price MLP with Adam on CPU."""

import csv
from pathlib import Path

import tensorflow as tf

from data_normalize import NORMALIZED_DIR
from data_split import TARGET


CPU_THREADS = 6
SEED = 42
HIDDEN_UNITS = 64
BATCH_SIZE = 256
LEARNING_RATE = 0.001
MAX_EPOCHS = 150
PATIENCE = 15
MODEL_PATH = Path(__file__).resolve().parent / "output_model" / "mlp_adam_model.keras"


def configure_cpu() -> None:
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(CPU_THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(1)


def read_training_csv(path: Path) -> tuple[list[str], tf.Tensor, tf.Tensor]:
    features: list[list[float]] = []
    prices: list[list[float]] = []
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        header = next(reader, [])
        if len(header) < 2 or header[-1] != TARGET:
            raise ValueError(f"{path.name} must end with a {TARGET} column.")

        price_column_index = len(header) - 1
        feature_columns: list[str] = []
        for column_index in range(price_column_index):
            feature_columns.append(header[column_index])

        for row in reader:
            if len(row) != len(header):
                raise ValueError(f"Wrong number of fields in {path.name} at line {reader.line_num}.")

            feature_row: list[float] = []
            for column_index in range(price_column_index):
                feature_row.append(float(row[column_index]))
            price = float(row[price_column_index])
            features.append(feature_row)
            prices.append([price])

    if not features:
        raise ValueError(f"{path.name} has no listings.")

    feature_tensor = tf.constant(features, dtype=tf.float32)
    price_tensor = tf.constant(prices, dtype=tf.float32)
    return feature_columns, feature_tensor, price_tensor


def print_scores(dataset_name: str, prices: tf.Tensor, predictions: tf.Tensor) -> None:
    if predictions.shape != prices.shape:
        raise ValueError("The model must predict one price per listing.")
    tf.debugging.assert_all_finite(predictions, "Price predictions must be finite.")

    errors = predictions - prices
    squared_errors = tf.square(errors)
    rmse = float(tf.sqrt(tf.reduce_mean(squared_errors)))
    absolute_errors = tf.abs(errors)
    mae = float(tf.reduce_mean(absolute_errors))

    r2_metric = tf.keras.metrics.R2Score()
    r2_metric.update_state(prices, predictions)
    r2 = float(r2_metric.result())

    allowed_errors = 0.10 * prices
    within_ten_percent = absolute_errors <= allowed_errors
    within_ten_percent_as_numbers = tf.cast(within_ten_percent, tf.float32)
    accuracy = float(tf.reduce_mean(within_ten_percent_as_numbers))

    print(f"{dataset_name} RMSE: {rmse:.4f} billion VND")
    print(f"{dataset_name} MAE: {mae:.4f} billion VND")
    print(f"{dataset_name} R2: {r2:.4f}")
    print(f"{dataset_name} within 10%: {accuracy:.2%}")


def train() -> None:
    configure_cpu()
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
    hidden_layer = tf.keras.layers.Dense(HIDDEN_UNITS, activation="relu")
    price_layer = tf.keras.layers.Dense(1)
    model = tf.keras.Sequential()
    model.add(input_layer)
    model.add(hidden_layer)
    model.add(price_layer)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_function = tf.keras.losses.MeanSquaredError()
    rmse_metric = tf.keras.metrics.RootMeanSquaredError(name="rmse")
    model.compile(optimizer=optimizer, loss=loss_function, metrics=[rmse_metric])

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

    rmse_by_epoch = history.history["val_rmse"]
    lowest_rmse = min(rmse_by_epoch)
    best_epoch = rmse_by_epoch.index(lowest_rmse) + 1
    print(f"Best epoch: {best_epoch} of {len(rmse_by_epoch)}")
    predictions = model(validation_features, training=False)
    print_scores("Validation", validation_prices, predictions)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Saved: {MODEL_PATH.name}")


if __name__ == "__main__":
    train()
