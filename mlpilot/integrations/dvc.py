"""DVC integration (needs ``dvc`` and ``git``; the ``[dvc]`` extra).

Version datasets with DVC when your team already uses it: the data goes to DVC's cache and
remote (S3, GCS, Azure, SSH, a shared drive), a small ``.dvc`` pointer goes into git, and any
past version can be read back by git revision.

>>> from mlpilot.integrations.dvc import DVCRepo, export_repository
>>> repo = DVCRepo.init("ml-data")                                  # git init + dvc init (or an existing repo)
>>> repo.add_remote("storage", "s3://bucket/dvc")                   # or a local / network folder
>>> v = repo.track(df, "data/customers.parquet", message="Q2 refresh", tag="customers-v2", push=True)
>>> repo.versions("data/customers.parquet")                         # rev, author, date, md5, size, tags
>>> old = repo.read("data/customers.parquet", rev="customers-v1")   # any git rev / tag / branch
>>> repo.log_to_run(run, "data/customers.parquet")                  # lineage: run -> exact data version
>>> export_repository(data_repo, "customers", repo)                 # replay DataRepository history into DVC

All operations shell out to the real ``git`` and ``dvc`` commands (the same ones you'd type), so
the result is an ordinary DVC repository that ``dvc pull`` / ``dvc diff`` / CI can use.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import io
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import polars as pl


class DVCError(RuntimeError):
    """A git or dvc command failed."""


@dataclasses.dataclass
class DVCVersion:
    path: str
    rev: str
    md5: str
    size: int | None
    message: str
    tags: list
    author: str | None = None
    date: str | None = None

    @property
    def ref(self) -> str:
        return f"{self.path}@{self.rev[:10]}#{self.md5[:12]}"


def _dvc_cmd() -> list:
    if importlib.util.find_spec("dvc") is not None:
        return [sys.executable, "-m", "dvc"]
    exe = shutil.which("dvc")
    if exe:
        return [exe]
    raise ImportError('The DVC integration needs dvc: pip install "mlpilot[dvc]"')


def _git_cmd() -> list:
    exe = shutil.which("git")
    if not exe:
        raise DVCError("git is not installed (DVC stores its pointers in git)")
    return [exe]


def _pointer(text: str) -> tuple:
    """(md5, size) from the contents of a ``.dvc`` file."""
    try:
        import yaml
        out = (yaml.safe_load(text) or {}).get("outs", [{}])[0]
        return str(out.get("md5", "")), out.get("size")
    except ImportError:                                  # pragma: no cover
        md5 = re.search(r"md5:\s*([0-9a-f]+)", text)
        size = re.search(r"size:\s*(\d+)", text)
        return (md5.group(1) if md5 else ""), (int(size.group(1)) if size else None)


class DVCRepo:
    """A git + DVC repository on disk."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        if not (self.root / ".dvc").is_dir():
            raise DVCError(f"{self.root} is not a DVC repository; use DVCRepo.init()")

    # ── commands ────────────────────────────────────────────────────────────
    def _run(self, cmd: list, *args, check: bool = True) -> str:
        env = {**os.environ, "DVC_NO_ANALYTICS": "1", "GIT_TERMINAL_PROMPT": "0"}
        p = subprocess.run([*cmd, *map(str, args)], cwd=self.root, capture_output=True, text=True, env=env,
                           encoding="utf-8", errors="replace")
        if check and p.returncode != 0:
            raise DVCError(f"{' '.join(map(str, [Path(cmd[-1]).name, *args]))} failed:\n{(p.stderr or p.stdout).strip()}")
        return p.stdout

    def git(self, *args, check: bool = True) -> str:
        ident = []
        if not self._run(_git_cmd(), "config", "user.email", check=False).strip():
            ident = ["-c", "user.name=mlpilot", "-c", "user.email=mlpilot@localhost"]
        return self._run(_git_cmd(), *ident, *args, check=check)

    def dvc(self, *args, check: bool = True) -> str:
        return self._run(_dvc_cmd(), *args, check=check)

    @classmethod
    def init(cls, root, *, git: bool = True) -> "DVCRepo":
        """Create (or reuse) a git + DVC repository at ``root``."""
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        if (root / ".dvc").is_dir():
            return cls(root)
        bare = object.__new__(cls)
        bare.root = root
        if git and not (root / ".git").exists():
            bare._run(_git_cmd(), "init", "-q")
        bare.dvc("init", "-q", *([] if git else ["--no-scm"]))
        bare.dvc("config", "core.analytics", "false")
        if git:
            bare.git("add", ".dvc", ".dvcignore")
            bare.git("commit", "-q", "-m", "Initialise DVC")
        return cls(root)

    # ── remotes ─────────────────────────────────────────────────────────────
    def add_remote(self, name: str, url: str, *, default: bool = True) -> None:
        """Register where data is pushed (``s3://``, ``gs://``, ``azure://``, ``ssh://`` or a folder)."""
        existing = self.dvc("remote", "list", check=False)
        if re.search(rf"(?m)^{re.escape(name)}\s", existing):
            self.dvc("remote", "modify", name, "url", url)
        else:
            self.dvc("remote", "add", *(["-d"] if default else []), name, url)
        if default:
            self.dvc("remote", "default", name)
        self.git("add", ".dvc/config")
        self.git("commit", "-q", "-m", f"DVC remote {name}", check=False)

    def push(self, remote: str | None = None, targets=()) -> None:
        self.dvc("push", "-q", *(["-r", remote] if remote else []), *targets)

    def pull(self, remote: str | None = None, targets=()) -> None:
        self.dvc("pull", "-q", *(["-r", remote] if remote else []), *targets)

    # ── tracking ────────────────────────────────────────────────────────────
    def _rel(self, path) -> str:
        p = Path(path)
        p = p if p.is_absolute() else self.root / p
        return p.resolve().relative_to(self.root).as_posix()

    def track(self, data, path, *, message: str | None = None, tag: str | None = None, push: bool = False,
              remote: str | None = None) -> DVCVersion:
        """Write ``data`` (a DataFrame, or an existing file to copy) to ``path`` in the repo, put
        it under DVC, and commit the pointer to git (optionally tagging the commit and pushing the
        data). If the data is identical to the tracked version, nothing new is committed."""
        rel = self._rel(path)
        dest = self.root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, (str, Path)):
            if Path(data).resolve() != dest.resolve():
                shutil.copy2(data, dest)
        else:
            df = data if isinstance(data, pl.DataFrame) else pl.from_pandas(data)
            _write(df, dest)
        self.dvc("add", "-q", rel)
        pointer, ignore = f"{rel}.dvc", (Path(rel).parent / ".gitignore").as_posix()
        self.git("add", pointer, *([ignore] if (self.root / ignore).exists() else []))
        if self.git("status", "--porcelain", "--", pointer, ignore, check=False).strip():
            self.git("commit", "-q", "-m", message or f"Update {rel}")
        if tag:
            self.git("tag", "-f", tag)
        if push:
            self.push(remote, [pointer])
        return self.version(rel)

    def version(self, path, rev: str = "HEAD") -> DVCVersion:
        """The tracked version of ``path`` at a git revision."""
        rel = self._rel(path)
        text = self.git("show", f"{rev}:{rel}.dvc", check=False)
        if not text:
            raise KeyError(f"{rel} is not tracked by DVC at {rev}")
        md5, size = _pointer(text)
        full = self.git("rev-list", "-n", "1", rev).strip()
        info = self.git("log", "-1", "--format=%an%x1f%aI%x1f%s", full).strip().split("\x1f")
        tags = [t for t in self.git("tag", "--points-at", full).split() if t]
        return DVCVersion(rel, full, md5, size, info[2] if len(info) > 2 else "", tags,
                          info[0] if info else None, info[1] if len(info) > 1 else None)

    def versions(self, path) -> pl.DataFrame:
        """Every committed version of ``path``, newest first."""
        rel = self._rel(path)
        revs = [r for r in self.git("log", "--format=%H", "--", f"{rel}.dvc").split() if r]
        rows = [dataclasses.asdict(self.version(rel, r)) for r in revs]
        for r in rows:
            r["tags"] = ",".join(r["tags"])
        schema = {"path": pl.String, "rev": pl.String, "md5": pl.String, "size": pl.Int64, "message": pl.String,
                  "tags": pl.String, "author": pl.String, "date": pl.String}
        return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)

    def read(self, path, rev: str | None = None, *, remote: str | None = None) -> pl.DataFrame:
        """A dataset as it was at git revision ``rev`` (a commit, tag or branch; default: the
        workspace). Data comes from the local DVC cache, or the remote if it isn't cached."""
        rel = self._rel(path)
        if rev is None and (self.root / rel).exists():
            return _read(self.root / rel)
        import dvc.api
        with dvc.api.open(rel, repo=str(self.root), rev=rev, remote=remote, mode="rb") as fh:
            return _read(io.BytesIO(fh.read()), suffix=Path(rel).suffix)

    def checkout(self, path, rev: str) -> Path:
        """Restore the workspace copy of ``path`` to the version at ``rev`` (fetched from the
        remote when it isn't in the local cache). Commit or check out the ``.dvc`` file again to
        go back."""
        rel = self._rel(path)
        self.git("checkout", rev, "--", f"{rel}.dvc")
        try:
            self.dvc("checkout", "--force", f"{rel}.dvc")
        except DVCError:                              # not cached locally: fetch it
            self.dvc("pull", "--force", f"{rel}.dvc")
        return self.root / rel

    def status(self) -> str:
        return self.dvc("status", check=False).strip()

    def log_to_run(self, run, path, rev: str = "HEAD") -> DVCVersion:
        """Record in a tracking run exactly which DVC data version it used."""
        v = self.version(path, rev)
        run.set_tag(f"dvc.{v.path}", f"{v.rev}:{v.md5}")
        try:
            run.log_dataset(self.read(v.path, None if rev == "HEAD" else rev), Path(v.path).stem, role="input")
        except RuntimeError:                         # run already ended: the tag is still recorded
            pass
        return v


def export_repository(data_repo, name: str, dvc_repo: DVCRepo, path=None, *, push: bool = False) -> list:
    """Replay the versions of dataset ``name`` from a :class:`~mlpilot.DataRepository`
    into DVC as one git commit per version, tagged ``<name>-v<N>`` (versions already exported
    are skipped). Returns the :class:`DVCVersion` list."""
    rel = dvc_repo._rel(path or f"data/{name}.parquet")
    done = set(dvc_repo.git("tag", "--list", f"{name}-v*").split())
    out = []
    for dv in sorted(data_repo._versions(name), key=lambda v: v.version):
        tag = f"{name}-v{dv.version}"
        if tag in done:
            out.append(dvc_repo.version(rel, tag))
            continue
        msg = f"{dv.ref}: {dv.message}" if dv.message else dv.ref
        out.append(dvc_repo.track(data_repo.root / "objects" / dv.file, rel, message=msg, tag=tag))
    if push:
        dvc_repo.push(targets=[f"{rel}.dvc"])
    return out


def _write(df: pl.DataFrame, dest: Path) -> None:
    ext = dest.suffix.lower()
    if ext == ".csv":
        df.write_csv(dest)
    elif ext in (".json", ".ndjson", ".jsonl"):
        df.write_ndjson(dest)
    else:
        df.write_parquet(dest, compression="zstd")


def _read(src, suffix: str | None = None) -> pl.DataFrame:
    ext = (suffix or Path(str(src)).suffix).lower()
    if ext == ".csv":
        return pl.read_csv(src)
    if ext in (".json", ".ndjson", ".jsonl"):
        return pl.read_ndjson(src)
    return pl.read_parquet(src)
