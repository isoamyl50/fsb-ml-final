"""Build repeatable train, validation, and test splits from the local CSV."""

import csv
import math
import unicodedata
from pathlib import Path
from random import Random
from statistics import median


HERE = Path(__file__).resolve().parent
DATASET_DIR = HERE / "dataset"
DATASET_PATH = DATASET_DIR / "raw" / "vietnam_housing_dataset.csv"
SPLIT_DIR = DATASET_DIR / "split"
TARGET = "Price"
FEATURE_COLUMNS = (
    "Address",
    "Area",
    "Frontage",
    "Access Road",
    "House direction",
    "Balcony direction",
    "Floors",
    "Bedrooms",
    "Bathrooms",
    "Legal status",
    "Furniture state",
)
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
RepostKey = tuple[str, str, str, str, str, str, str, str, str, str, str]


def repost_key(row: dict[str, str]) -> RepostKey:
    address = unicodedata.normalize("NFC", row["Address"])
    address = " ".join(address.split()).rstrip(".").strip()
    return (
        address,
        row["Area"].strip(),
        row["Frontage"].strip(),
        row["Access Road"].strip(),
        row["House direction"].strip(),
        row["Balcony direction"].strip(),
        row["Floors"].strip(),
        row["Bedrooms"].strip(),
        row["Bathrooms"].strip(),
        row["Legal status"].strip(),
        row["Furniture state"].strip(),
    )


def group_median_price(group: list[tuple[dict[str, str], float]]) -> float:
    prices: list[float] = []
    for listing in group:
        price = listing[1]
        prices.append(price)
    return median(prices)


def load_splits(
    dataset_path: Path = DATASET_PATH,
    seed: int = 42,
) -> dict[str, tuple[list[dict[str, str]], list[float]]]:
    """Return (X, y) for each partition, with prices in billion VND."""
    with dataset_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required: set[str] = set(FEATURE_COLUMNS)
        required.add(TARGET)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
        groups: dict[RepostKey, list[tuple[dict[str, str], float]]] = {}
        for line, row in enumerate(reader, start=2):
            if None in row or None in row.values():
                raise ValueError(f"Unexpected number of fields at CSV row {line}.")
            features: dict[str, str] = {}
            for column in FEATURE_COLUMNS:
                features[column] = row[column]
            key = repost_key(features)
            price = float(row[TARGET])
            if not math.isfinite(price):
                raise ValueError(f"Invalid price at CSV row {line}.")
            if key not in groups:
                groups[key] = []
            groups[key].append((features, price))

    # Rank groups by median price so each partition gets low- and high-priced listings.
    ordered = sorted(groups.values(), key=group_median_price)
    rng = Random(seed)
    splits: dict[str, tuple[list[dict[str, str]], list[float]]] = {}
    for name in SPLIT_RATIOS:
        splits[name] = ([], [])

    band_count = 10
    for band_index in range(band_count):
        start = band_index * len(ordered) // band_count
        end = (band_index + 1) * len(ordered) // band_count
        band = ordered[start:end]
        rng.shuffle(band)
        band_size = 0
        for group in band:
            band_size += len(group)

        remaining: dict[str, float] = {}
        for name, ratio in SPLIT_RATIOS.items():
            remaining[name] = band_size * ratio

        for group in band:
            # Assign the entire group to the split furthest below its target row count.
            chosen_split = "train"
            for name in remaining:
                if remaining[name] > remaining[chosen_split]:
                    chosen_split = name

            split_features, split_prices = splits[chosen_split]
            for row, price in group:
                split_features.append(row)
                split_prices.append(price)
            remaining[chosen_split] -= len(group)

    for name, split in splits.items():
        split_prices = split[1]
        if not split_prices:
            raise ValueError(f"The {name} split has no listings.")
    return splits


if __name__ == "__main__":
    columns: list[str] = list(FEATURE_COLUMNS)
    columns.append(TARGET)
    SPLIT_DIR.mkdir(exist_ok=True)
    for name, (features, prices) in load_splits().items():
        output_path = SPLIT_DIR / f"{name}.csv"
        with output_path.open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            for row, price in zip(features, prices, strict=True):
                output_row = row.copy()
                output_row[TARGET] = str(price)
                writer.writerow(output_row)
        print(f"{output_path.name}: {len(features):,} listings")
