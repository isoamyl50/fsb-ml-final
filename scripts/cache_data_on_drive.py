"""Colab VM setup & verification script: inspect GPU/packages, verify Drive FUSE mount, and cache CSV."""

from __future__ import annotations

import csv
import importlib.metadata
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

DRIVE_MOUNT_ROOT = Path("/content/drive")
DRIVE_MYDRIVE = DRIVE_MOUNT_ROOT / "MyDrive"
DRIVE_PROJECT_DIR = DRIVE_MYDRIVE / "ml-final-project"
DRIVE_DATA_DIR = DRIVE_PROJECT_DIR / "data"
DRIVE_ARTIFACTS_DIR = DRIVE_PROJECT_DIR / "artifacts"

LOCAL_UPLOAD_CSV = Path("/content/vietnam_housing_dataset.csv")
CACHED_CSV = DRIVE_DATA_DIR / "vietnam_housing_dataset.csv"

EXPECTED_BYTES = 3_532_685
EXPECTED_ROWS = 30_229

PACKAGES_TO_CHECK = [
    "numpy",
    "pandas",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "xgboost",
    "lightgbm",
    "catboost",
    "optuna",
    "torch",
]


def print_runtime_info() -> None:
    """Print Python, GPU, CUDA, and installed ML package versions."""
    print(f"Python: {sys.version.split()[0]}")
    try:
        import torch

        print(f"PyTorch: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024 ** 3)
            print(f"GPU Device: {torch.cuda.get_device_name(0)} ({vram_gb:.2f} GB VRAM)")
    except Exception as exc:
        print(f"PyTorch check failed: {exc}")

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip()
        print(f"nvidia-smi: {out}")
    except Exception as exc:
        print(f"nvidia-smi unavailable: {exc}")

    print("\nInstalled ML packages:")
    for pkg in PACKAGES_TO_CHECK:
        try:
            ver = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            ver = "not installed"
        print(f"  {pkg:15s}: {ver}")


def verify_drive_fuse_mount() -> None:
    """Ensure /content/drive is genuinely FUSE-mounted via /proc/mounts."""
    mounts_text = Path("/proc/mounts").read_text(encoding="utf-8", errors="replace")
    drive_mounts = [
        line for line in mounts_text.splitlines() if " /content/drive" in line
    ]
    if not drive_mounts or not DRIVE_MYDRIVE.is_dir():
        raise RuntimeError(
            "Google Drive FUSE is not mounted at /content/drive. "
            f"Found mount lines: {drive_mounts}"
        )
    print("\nGoogle Drive FUSE mount verified:")
    for line in drive_mounts:
        print(f"  {line}")


def cache_and_verify_dataset() -> None:
    """Cache vietnam_housing_dataset.csv into Google Drive and verify size/row integrity."""
    DRIVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DRIVE_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    needs_copy = not CACHED_CSV.exists() or CACHED_CSV.stat().st_size != EXPECTED_BYTES
    if needs_copy:
        if not LOCAL_UPLOAD_CSV.exists():
            raise FileNotFoundError(
                f"Cached file missing at {CACHED_CSV} and upload not found at {LOCAL_UPLOAD_CSV}"
            )
        shutil.copy2(LOCAL_UPLOAD_CSV, CACHED_CSV)
        os.sync()
        time.sleep(2.0)
        print(f"\n[cache] Copied {LOCAL_UPLOAD_CSV} -> {CACHED_CSV}")
    else:
        print(f"\n[cache] Dataset already cached with expected size at {CACHED_CSV}")

    actual_bytes = CACHED_CSV.stat().st_size
    if actual_bytes != EXPECTED_BYTES:
        raise AssertionError(
            f"Size mismatch at {CACHED_CSV}: expected {EXPECTED_BYTES}, got {actual_bytes}"
        )

    with CACHED_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        cols = list(reader.fieldnames or [])
        row_count = sum(1 for _ in reader)

    if row_count != EXPECTED_ROWS:
        raise AssertionError(
            f"Row count mismatch at {CACHED_CSV}: expected {EXPECTED_ROWS}, got {row_count}"
        )

    if LOCAL_UPLOAD_CSV.exists():
        LOCAL_UPLOAD_CSV.unlink()
        print(f"[cache] Removed temporary VM copy {LOCAL_UPLOAD_CSV}")

    print(f"[verified] Path   : {CACHED_CSV}")
    print(f"[verified] Size   : {actual_bytes:,} bytes")
    print(f"[verified] Rows   : {row_count:,}")
    print(f"[verified] Columns: {cols}")


if __name__ == "__main__":
    print_runtime_info()
    verify_drive_fuse_mount()
    cache_and_verify_dataset()
