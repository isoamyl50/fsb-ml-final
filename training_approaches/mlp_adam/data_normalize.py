"""Fit preprocessing on Train and apply it to raw listing inputs."""

import csv
import math
import unicodedata
from pathlib import Path
from statistics import mean, median, pstdev

from data_split import DATASET_DIR, FEATURE_COLUMNS, SPLIT_DIR, TARGET


NUMERIC_COLUMNS = ("Area", "Frontage", "Access Road", "Floors", "Bedrooms", "Bathrooms")
CATEGORY_COLUMNS = ("House direction", "Balcony direction", "Legal status", "Furniture state")
NORMALIZED_DIR = DATASET_DIR / "normalized"


def read_split(path: Path) -> tuple[list[dict[str, str]], list[float]]:
    rows: list[dict[str, str]] = []
    prices: list[float] = []
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required: set[str] = set(FEATURE_COLUMNS)
        required.add(TARGET)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name} is missing columns: {', '.join(sorted(missing))}")

        for row in reader:
            if None in row or None in row.values():
                raise ValueError(f"Wrong number of fields in {path.name} at line {reader.line_num}.")
            price = float(row.pop(TARGET))
            if not math.isfinite(price):
                raise ValueError(f"Invalid Price in {path.name} at line {reader.line_num}.")
            rows.append(row)
            prices.append(price)
    return rows, prices


def numeric_value(row: dict[str, str], column: str) -> float | None:
    text = row[column].strip()
    if not text:
        return None
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"Invalid value in {column}: {text!r}")
    return value


def category_values(row: dict[str, str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for column in CATEGORY_COLUMNS:
        values[column] = unicodedata.normalize("NFC", row[column].strip())

    address = unicodedata.normalize("NFC", row["Address"])
    address = " ".join(address.split()).rstrip(".").strip()
    parts = address.rsplit(",", maxsplit=2)
    province = ""
    district = ""
    if len(parts) >= 2:
        province = parts[-1].strip()
        district = parts[-2].strip()
    if province == "TP. HCM":
        province = "Hồ Chí Minh"

    values["Province"] = province
    values["District"] = ""
    if province and district:
        values["District"] = f"{district}, {province}"
    return values


class DataNormalizer:
    def __init__(self, train_rows: list[dict[str, str]]) -> None:
        if not train_rows:
            raise ValueError("Train has no listings.")

        self.medians: dict[str, float] = {}
        self.means: dict[str, float] = {}
        self.standard_deviations: dict[str, float] = {}
        self.category_indexes: dict[str, dict[str, int]] = {}
        self.feature_names: list[str] = []

        for column in NUMERIC_COLUMNS:
            observed: list[float] = []
            for row in train_rows:
                value = numeric_value(row, column)
                if value is not None:
                    observed.append(value)
            if not observed:
                raise ValueError(f"Train has no numeric values for {column}.")

            fill = median(observed)
            missing_count = len(train_rows) - len(observed)
            for _ in range(missing_count):
                observed.append(fill)
            self.medians[column] = fill
            self.means[column] = mean(observed)
            self.standard_deviations[column] = pstdev(observed) or 1.0
            self.feature_names.append(column)
            self.feature_names.append(f"{column}_missing")

        categories: dict[str, set[str]] = {}
        for row in train_rows:
            for column, value in category_values(row).items():
                if column not in categories:
                    categories[column] = {""}
                categories[column].add(value)

        for column, values in categories.items():
            indexes: dict[str, int] = {}
            for value in sorted(values):
                indexes[value] = len(self.feature_names)
                label = value or "(missing)"
                self.feature_names.append(f"{column}={label}")
            self.category_indexes[column] = indexes

    def transform(self, rows: list[dict[str, str]]) -> list[list[float]]:
        matrix: list[list[float]] = []
        for row in rows:
            features: list[float] = []
            for column in NUMERIC_COLUMNS:
                value = numeric_value(row, column)
                is_missing = 0.0
                if value is None:
                    value = self.medians[column]
                    is_missing = 1.0
                standardized = (value - self.means[column]) / self.standard_deviations[column]
                features.append(standardized)
                features.append(is_missing)

            while len(features) < len(self.feature_names):
                features.append(0.0)
            for column, value in category_values(row).items():
                index = self.category_indexes[column].get(value)
                if index is not None:
                    features[index] = 1.0
            matrix.append(features)
        return matrix


if __name__ == "__main__":
    train_rows, train_prices = read_split(SPLIT_DIR / "train.csv")
    validation_rows, validation_prices = read_split(SPLIT_DIR / "validation.csv")
    normalizer = DataNormalizer(train_rows)

    NORMALIZED_DIR.mkdir(exist_ok=True)
    columns = normalizer.feature_names.copy()
    columns.append(TARGET)
    partitions = (
        ("train", train_rows, train_prices),
        ("validation", validation_rows, validation_prices),
    )
    for name, rows, prices in partitions:
        matrix = normalizer.transform(rows)
        output_path = NORMALIZED_DIR / f"{name}.csv"
        with output_path.open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(columns)
            for features, price in zip(matrix, prices, strict=True):
                features.append(price)
                writer.writerow(features)
        print(f"normalized/{output_path.name}: {len(rows):,} rows, {len(columns) - 1} features")
