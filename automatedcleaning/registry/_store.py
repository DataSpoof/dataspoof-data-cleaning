"""Small file-store helpers: atomic writes, a cross-process lock, names, users and git info."""

from __future__ import annotations

import datetime as _dt
import getpass
import hashlib
import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path

from ..quality.checks import jsonable

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,99}$")


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def user() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


def host() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"


def check_name(name: str, what: str = "name") -> str:
    """Names become directory names, so only letters, digits, ``_ . -`` (no path tricks)."""
    if not isinstance(name, str) or not _NAME.match(name) or name in (".", ".."):
        raise ValueError(f"Invalid {what} {name!r}: use letters, digits, '_', '.', '-' (max 100 chars)")
    return name


def write_json(path: Path, data) -> None:
    """Write JSON atomically (temp file + rename), so readers never see half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(jsonable(data), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(jsonable(record), ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: Path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue                     # a torn last line from a crash; skip it
    return out


class FileLock:
    """Exclusive lock via an ``O_EXCL`` lock file (works across processes on one machine
    or a shared drive). Locks older than ``stale`` seconds are assumed abandoned."""

    def __init__(self, path: Path, timeout: float = 30.0, stale: float = 120.0):
        self.path, self.timeout, self.stale = Path(path), timeout, stale

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.getpid()}@{host()}".encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    if time.time() - self.path.stat().st_mtime > self.stale:
                        self.path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Could not acquire lock {self.path}")
                time.sleep(0.05)

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


def git_info(cwd=None) -> dict | None:
    """Current git commit / branch / dirty flag, if run inside a repository."""
    try:
        def git(*args):
            return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                                  timeout=5).stdout.strip()
        commit = git("rev-parse", "HEAD")
        if not commit:
            return None
        return {"commit": commit, "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
    except Exception:
        return None


def dataset_fingerprint(df, name: str | None = None) -> dict:
    """Identify a dataset version: shape, schema, and a content hash.

    The content hash covers every value (row order matters). It is stable for the same data
    within one Polars version; the schema hash is stable everywhere."""
    import polars as pl
    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)
    schema = {c: str(t) for c, t in df.schema.items()}
    schema_hash = hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()[:16]
    digest = hashlib.sha256(schema_hash.encode())
    if df.height and df.width:
        digest.update(df.hash_rows(seed=0).to_numpy().tobytes())
    return {"name": name, "rows": df.height, "columns": df.width, "schema": schema,
            "schema_hash": schema_hash, "content_hash": digest.hexdigest()[:32],
            "null_cells": int(sum(df.null_count().row(0))) if df.width else 0}
