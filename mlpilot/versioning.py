"""Data version control: commit, tag, diff and restore datasets — like git, for DataFrames.

>>> repo = DataRepository("data_repo")                      # a folder (or a shared drive)
>>> v1 = repo.commit(df, "customers", message="initial extract")
>>> v2 = repo.commit(df2, "customers", message="Q2 refresh", tags=["gold"])
>>> repo.log("customers")                                   # every version: who, when, rows, hash
>>> repo.diff("customers", 1, 2, key="customer_id")         # schema changes + added/removed/changed rows
>>> old = repo.checkout("customers", version=1)             # exact bytes back (checksum-verified)
>>> repo.checkout("customers", tag="gold")

Each version is stored once as compressed Parquet under its content hash, so committing the
same data again doesn't duplicate it, and any tampering with a stored file is detected on
checkout. Commits record the author, message, parent, tags, optional source/lineage and the
tracking run that produced or used it (``run=``).
"""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path

import polars as pl

from .automl._persist import file_sha256
from .registry._store import FileLock, check_name, dataset_fingerprint, now, read_json, user, write_json


@dataclasses.dataclass
class DatasetVersion:
    name: str
    version: int
    content_hash: str
    schema_hash: str
    rows: int
    columns: int
    schema: dict
    created_at: str
    author: str
    message: str
    parent: int | None
    tags: list
    source: str | None
    run_id: str | None
    file: str
    bytes: int
    file_sha256: str = ""

    @property
    def ref(self) -> str:
        """``name@vN#hash`` - a stable reference to log with experiments."""
        return f"{self.name}@v{self.version}#{self.content_hash[:12]}"

    def __repr__(self) -> str:
        return f"DatasetVersion({self.ref}, {self.rows:,} rows x {self.columns}, '{self.message}')"


@dataclasses.dataclass
class DatasetDiff:
    name: str
    a: int
    b: int
    rows_a: int
    rows_b: int
    added_columns: list
    removed_columns: list
    changed_types: dict
    added_rows: int | None = None
    removed_rows: int | None = None
    changed_rows: int | None = None
    unchanged_rows: int | None = None
    changed_columns: dict = dataclasses.field(default_factory=dict)

    @property
    def identical(self) -> bool:
        return not (self.added_columns or self.removed_columns or self.changed_types) and \
            self.rows_a == self.rows_b and not self.added_rows and not self.removed_rows and not self.changed_rows

    def summary(self) -> str:
        lines = [f"{self.name}: v{self.a} -> v{self.b}: {self.rows_a:,} -> {self.rows_b:,} rows"]
        if self.added_columns:
            lines.append(f"  + columns: {self.added_columns}")
        if self.removed_columns:
            lines.append(f"  - columns: {self.removed_columns}")
        if self.changed_types:
            lines.append(f"  ~ types: {self.changed_types}")
        if self.added_rows is not None:
            lines.append(f"  rows: +{self.added_rows:,} added, -{self.removed_rows:,} removed, "
                         f"~{self.changed_rows:,} changed, {self.unchanged_rows:,} unchanged")
            if self.changed_columns:
                lines.append("  most changed columns: " + ", ".join(f"{c} ({n})" for c, n in
                                                                    list(self.changed_columns.items())[:8]))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


class DataRepository:
    def __init__(self, root="data_repo"):
        self.root = Path(root)
        (self.root / "objects").mkdir(parents=True, exist_ok=True)

    def _meta_dir(self, name: str) -> Path:
        return self.root / "datasets" / check_name(name, "dataset name")

    def _versions(self, name: str) -> list:
        d = self._meta_dir(name)
        files = sorted(d.glob("v*.json"), key=lambda p: int(p.stem[1:])) if d.exists() else []
        return [DatasetVersion(**read_json(f)) for f in files]

    # ── writing ─────────────────────────────────────────────────────────────
    def commit(self, df, name: str, *, message: str = "", tags=None, source: str | None = None, run=None,
               allow_unchanged: bool = False) -> DatasetVersion:
        """Store ``df`` as the next version of ``name``. If it is identical to the latest version,
        that version is returned instead (unless ``allow_unchanged``)."""
        df = df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
        fp = dataset_fingerprint(df, name)
        with FileLock(self.root / ".lock"):
            existing = self._versions(name)
            if existing and existing[-1].content_hash == fp["content_hash"] and not allow_unchanged:
                return existing[-1]
            obj = self.root / "objects" / f"{fp['content_hash']}.parquet"
            if not obj.exists():
                tmp = obj.with_suffix(".partial")
                df.write_parquet(tmp, compression="zstd", statistics=True)
                tmp.replace(obj)
            version = (existing[-1].version + 1) if existing else 1
            dv = DatasetVersion(name=name, version=version, content_hash=fp["content_hash"], schema_hash=fp["schema_hash"],
                                rows=df.height, columns=df.width, schema=fp["schema"], created_at=now(), author=user(),
                                message=message, parent=existing[-1].version if existing else None,
                                tags=sorted(set(tags or [])), source=source, run_id=getattr(run, "id", None),
                                file=obj.name, bytes=obj.stat().st_size, file_sha256=file_sha256(obj))
            if dv.tags:                                 # a tag points at one version: move it here
                for old in existing:
                    if set(old.tags) & set(dv.tags):
                        old.tags = sorted(set(old.tags) - set(dv.tags))
                        write_json(self._meta_dir(name) / f"v{old.version}.json", dataclasses.asdict(old))
            write_json(self._meta_dir(name) / f"v{version}.json", dataclasses.asdict(dv))
        if run is not None:
            run.log_dataset(df, name, role="output")
            run.set_tag(f"dataset.{name}", dv.ref)
        return dv

    def tag(self, name: str, version: int, tag: str) -> DatasetVersion:
        """Point ``tag`` (e.g. ``"gold"``, ``"training-2024Q2"``) at ``version`` (moved if it existed)."""
        with FileLock(self.root / ".lock"):
            target = None
            for v in self._versions(name):
                changed = False
                if tag in v.tags and v.version != version:
                    v.tags = [t for t in v.tags if t != tag]
                    changed = True
                if v.version == version:
                    v.tags = sorted(set(v.tags) | {tag})
                    target, changed = v, True
                if changed:
                    write_json(self._meta_dir(name) / f"v{v.version}.json", dataclasses.asdict(v))
            if target is None:
                raise KeyError(f"{name} has no version {version}")
        return target

    # ── reading ─────────────────────────────────────────────────────────────
    def get(self, name: str, version: int | None = None, tag: str | None = None) -> DatasetVersion:
        versions = self._versions(name)
        if not versions:
            raise KeyError(f"No dataset named {name!r}")
        if tag is not None:
            hit = [v for v in versions if tag in v.tags]
            if not hit:
                raise KeyError(f"{name} has no version tagged {tag!r}")
            return hit[-1]
        if version is None:
            return versions[-1]
        for v in versions:
            if v.version == int(version):
                return v
        raise KeyError(f"{name} has no version {version}")

    def checkout(self, name: str, version: int | None = None, *, tag: str | None = None, run=None,
                 verify: bool = True, lazy: bool = False):
        """The exact data of a version (latest by default). The content hash is re-checked, so a
        modified or corrupted stored file raises instead of silently returning different data."""
        dv = self.get(name, version, tag)
        path = self.root / "objects" / dv.file
        if lazy:
            return pl.scan_parquet(path)
        if verify and dv.file_sha256 and file_sha256(path) != dv.file_sha256:
            # the file's SHA-256 is stable across library versions (row hashes are not)
            raise ValueError(f"{dv.ref}: stored data does not match its checksum (modified or corrupted)")
        df = pl.read_parquet(path)
        if run is not None:
            run.log_dataset(df, name, role="input")
            run.set_tag(f"dataset.{name}", dv.ref)
        return df

    def log(self, name: str) -> pl.DataFrame:
        rows = [{"version": v.version, "created_at": v.created_at, "author": v.author, "message": v.message,
                 "rows": v.rows, "columns": v.columns, "tags": ",".join(v.tags), "content_hash": v.content_hash[:12],
                 "bytes": v.bytes, "run_id": v.run_id, "source": v.source} for v in self._versions(name)]
        if not rows:
            raise KeyError(f"No dataset named {name!r}")
        return pl.DataFrame(rows, infer_schema_length=None).sort("version", descending=True)

    def datasets(self) -> pl.DataFrame:
        base = self.root / "datasets"
        rows = []
        for d in sorted(p for p in base.iterdir() if p.is_dir()) if base.exists() else []:
            vs = self._versions(d.name)
            if vs:
                rows.append({"name": d.name, "versions": len(vs), "latest": vs[-1].version, "rows": vs[-1].rows,
                             "updated_at": vs[-1].created_at, "tags": ",".join(t for v in vs for t in v.tags)})
        schema = {"name": pl.String, "versions": pl.Int64, "latest": pl.Int64, "rows": pl.Int64, "updated_at": pl.String,
                  "tags": pl.String}
        return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)

    def diff(self, name: str, a: int, b: int, *, key=None) -> DatasetDiff:
        """Compare two versions: schema changes and, with ``key`` (column(s) identifying a row),
        rows added / removed / changed and which columns changed most. Without a key, rows are
        compared as whole records."""
        va, vb = self.get(name, a), self.get(name, b)
        da, db = self.checkout(name, a), self.checkout(name, b)
        sa, sb = va.schema, vb.schema
        out = DatasetDiff(name, a, b, da.height, db.height, [c for c in sb if c not in sa], [c for c in sa if c not in sb],
                          {c: f"{sa[c]} -> {sb[c]}" for c in sa if c in sb and sa[c] != sb[c]})
        common = [c for c in sa if c in sb]
        if key is not None:
            keys = [key] if isinstance(key, str) else list(key)
            left = da.select(common).with_columns(pl.lit(True).alias("_in_a"))
            right = db.select(common).with_columns(pl.lit(True).alias("_in_b"))
            j = left.join(right, on=keys, how="full", coalesce=True, suffix="__b")
            out.added_rows = int(j.filter(pl.col("_in_a").is_null()).height)
            out.removed_rows = int(j.filter(pl.col("_in_b").is_null()).height)
            both = j.filter(pl.col("_in_a").is_not_null() & pl.col("_in_b").is_not_null())
            changes = {}
            diff_any = pl.lit(False)
            for c in [c for c in common if c not in keys]:
                ne = pl.col(c).cast(pl.String).ne_missing(pl.col(f"{c}__b").cast(pl.String))
                n = int(both.select(ne.sum()).item())
                if n:
                    changes[c] = n
                diff_any = diff_any | ne
            out.changed_rows = int(both.select(diff_any.sum()).item()) if common else 0
            out.unchanged_rows = both.height - out.changed_rows
            out.changed_columns = dict(sorted(changes.items(), key=lambda kv: -kv[1]))
        else:
            ha = da.select(common).hash_rows(seed=0).value_counts()
            hb = db.select(common).hash_rows(seed=0).value_counts()
            ca = dict(ha.iter_rows())
            cb = dict(hb.iter_rows())
            out.added_rows = sum(max(0, n - ca.get(h, 0)) for h, n in cb.items())
            out.removed_rows = sum(max(0, n - cb.get(h, 0)) for h, n in ca.items())
            out.changed_rows = 0
            out.unchanged_rows = sum(min(n, cb.get(h, 0)) for h, n in ca.items())
        return out

    def gc(self) -> int:
        """Delete stored objects no version refers to any more; returns how many were removed."""
        used = {v.file for d in (self.root / "datasets").glob("*") if d.is_dir() for v in self._versions(d.name)}
        removed = 0
        for obj in (self.root / "objects").glob("*.parquet"):
            if obj.name not in used:
                obj.unlink()
                removed += 1
        return removed

    def export(self, name: str, version: int, dest) -> Path:
        """Copy a version's Parquet file out (e.g. to hand to another team)."""
        dv = self.get(name, version)
        return Path(shutil.copy2(self.root / "objects" / dv.file, dest))
