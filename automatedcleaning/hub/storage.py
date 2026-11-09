"""Artifact storage for the Hub: a local/NFS directory, or object storage via fsspec
(``s3://bucket/prefix`` with s3fs, ``gs://`` with gcsfs, ``az://`` / ``abfs://`` with adlfs).

Artifacts are stored as immutable ``.tar.gz`` blobs keyed by workspace / model / version, each
with its SHA-256 recorded in the database and re-checked on download."""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path, PurePosixPath


def tar_directory(path) -> bytes:
    """A model directory as a deterministic .tar.gz (sorted entries, no owner/time metadata)."""
    path = Path(path)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=6) as tar:
        for f in sorted(p for p in path.rglob("*") if p.is_file()):
            info = tarfile.TarInfo(f.relative_to(path).as_posix())
            data = f.read_bytes()
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def untar_to(blob: bytes, dest) -> Path:
    """Extract an artifact safely: only regular files, no absolute paths, no '..' (zip-slip)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            target = (dest / m.name).resolve()
            if root not in target.parents:
                raise ValueError(f"Unsafe path in artifact: {m.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(m)
            target.write_bytes(src.read() if src else b"")
    return dest


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ArtifactStore:
    """``ArtifactStore("/var/lib/hub/artifacts")`` or ``ArtifactStore("s3://ml-artifacts/hub")``."""

    def __init__(self, root: str):
        self.root = str(root).rstrip("/")
        self.remote = "://" in self.root and not self.root.startswith("file://")
        if self.remote:
            try:
                import fsspec
            except ImportError as exc:
                raise ImportError('Object-storage artifacts need fsspec (+ s3fs / gcsfs / adlfs): '
                                  'pip install "AutomatedCleaning[server]" s3fs') from exc
            self.fs, self.base = fsspec.core.url_to_fs(self.root)
        else:
            self.fs, self.base = None, str(Path(self.root.removeprefix("file://")).resolve())
            Path(self.base).mkdir(parents=True, exist_ok=True)

    def _key(self, *parts) -> str:
        for p in parts:
            if not p or "/" in str(p) or "\\" in str(p) or str(p) in (".", ".."):
                raise ValueError(f"Invalid artifact key part {p!r}")
        return str(PurePosixPath(*map(str, parts)))

    def put(self, blob: bytes, *parts) -> str:
        key = self._key(*parts)
        if self.fs is not None:
            path = f"{self.base}/{key}"
            if self.fs.exists(path):
                raise FileExistsError(f"Artifact {key} already exists (artifacts are immutable)")
            with self.fs.open(path, "wb") as fh:
                fh.write(blob)
            return f"{self.root}/{key}"
        local = Path(self.base) / key
        if local.exists():
            raise FileExistsError(f"Artifact {key} already exists (artifacts are immutable)")
        local.parent.mkdir(parents=True, exist_ok=True)
        tmp = local.with_suffix(".partial")
        tmp.write_bytes(blob)
        tmp.replace(local)
        return local.as_uri()

    def get(self, uri: str) -> bytes:
        if self.fs is not None:
            if not uri.startswith(self.root + "/"):
                raise ValueError("Artifact is outside this store")
            with self.fs.open(f"{self.base}/{uri[len(self.root) + 1:]}", "rb") as fh:
                return fh.read()
        from urllib.parse import urlparse
        from urllib.request import url2pathname
        p = Path(url2pathname(urlparse(uri).path)) if uri.startswith("file:") else Path(uri)
        resolved = p.resolve()
        if Path(self.base) not in resolved.parents:
            raise ValueError("Artifact is outside this store")
        return resolved.read_bytes()
