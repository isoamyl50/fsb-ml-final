"""Atomic output and cooperative file locks for experiment writers."""

import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def file_lock(path, timeout=0):
    """Reserve a path exclusively; never delete another process's lock.

    A killed process can leave its lock behind. Do not steal locks based on age.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise FileExistsError(f"Experiment output is reserved: {path}") from None
            time.sleep(0.05)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(dict(pid=os.getpid()), fh)
        yield
    finally:
        path.unlink()


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp.{uuid.uuid4().hex}")
    try:
        tmp.write_text(text, encoding="utf-8", newline="")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
