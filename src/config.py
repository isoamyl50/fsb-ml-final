"""Canonical paths, constants, and environment-aware dataset resolution for regression."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_PATH = DATA_DIR / "raw" / "vietnam_housing_dataset.csv"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
EXPERIMENTS_CONFIGS_DIR = EXPERIMENTS_DIR / "configs"
EXPERIMENTS_RUNS_DIR = EXPERIMENTS_DIR / "runs"
EXPERIMENTS_SPLITS_PATH = EXPERIMENTS_DIR / "splits" / "splits_v1.json"
LEADERBOARD_MD_PATH = EXPERIMENTS_DIR / "leaderboard.md"
LEADERBOARD_CSV_PATH = EXPERIMENTS_DIR / "leaderboard.csv"

SLIDES_DIR = PROJECT_ROOT / "slides"
SLIDES_SCRIPTS_DIR = SLIDES_DIR / "scripts"
SLIDES_FIGURES_DIR = SLIDES_DIR / "public" / "figures"
SLIDES_RESULTS_PATH = SLIDES_DIR / "data" / "results.json"

DRIVE_MOUNT_ROOT = Path("/content/drive")
DRIVE_DATA_PATH = (
    DRIVE_MOUNT_ROOT / "MyDrive" / "ml-final-project" / "data" / "vietnam_housing_dataset.csv"
)
DRIVE_RUNS_DIR = DRIVE_MOUNT_ROOT / "MyDrive" / "ml-final-project" / "runs"
VM_DATA_PATH = Path("/content/vietnam_housing_dataset.csv")

RANDOM_SEED = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # Train / Validation / Test

RAW_NUM_COLS = [
    "Area",
    "Frontage",
    "Access Road",
    "Floors",
    "Bedrooms",
    "Bathrooms",
]
RAW_CAT_COLS = [
    "House direction",
    "Balcony direction",
    "Legal status",
    "Furniture state",
]
ADDRESS_COL = "Address"
TARGET_COL = "Price"

DEFAULT_TARGET_TRANSFORM = "raw"
PRICE_MIN_CLIP = 0.5
PRICE_MAX_CLIP = 15.0


def resolve_dataset_path(explicit_path: str | Path | None = None) -> Path:
    """Resolve dataset CSV path across local workspace and Google Colab / Drive."""
    if explicit_path is not None:
        candidate = Path(explicit_path)
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Explicit dataset path not found: {candidate}")

    for candidate in (DATA_RAW_PATH, DRIVE_DATA_PATH, VM_DATA_PATH):
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Could not locate vietnam_housing_dataset.csv in any expected path: "
        f"{DATA_RAW_PATH}, {DRIVE_DATA_PATH}, {VM_DATA_PATH}"
    )
