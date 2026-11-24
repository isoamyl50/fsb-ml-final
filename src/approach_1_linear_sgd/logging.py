"""Publish approach results into the shared experiment schema without retraining."""

import copy
import json
import re
from pathlib import Path

from ..experiment_io import file_lock
from ..runner import _atomic_write_json


def run_lock_path(project_root, run_id):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", run_id):
        raise ValueError("Invalid run_id")
    return Path(project_root) / "experiments" / ".locks" / f"{run_id}.lock"


def publish_shared_run(payload, project_root):
    """Caller must hold this run's lock for the entire read/check/write operation."""
    run_id = payload["run_id"]
    run_lock_path(project_root, run_id)  # Validate before constructing file paths.
    shared = copy.deepcopy(payload)
    shared.pop("saved_path", None)
    shared.pop("shared_saved_path", None)
    shared.update(task="regression", outlier_filter=False)
    shared["features"]["preserve_nans"] = False
    path = Path(project_root) / "experiments" / "runs" / f"{run_id}.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != shared:
            raise FileExistsError(f"Refusing to overwrite a different shared result: {path}")
    else:
        _atomic_write_json(path, shared)
    return path


def sync_existing_run(path, project_root):
    """Idempotently export an existing run; never fit or evaluate a model."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("approach") != "approach_1_linear_sgd":
        raise ValueError(f"Not an approach 1 result: {path}")
    with file_lock(run_lock_path(project_root, payload["run_id"])):
        return publish_shared_run(payload, project_root)
