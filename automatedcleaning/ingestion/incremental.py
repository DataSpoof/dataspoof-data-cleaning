"""Incremental loads: only the rows (or files) that are new since the last successful load.

Modes:

* **cursor** — a database table (or any callable) with a monotonically increasing column such
  as ``updated_at`` or an auto-increment id. Each load asks for rows past the saved watermark
  (as a bound parameter). With ``key=``, rows that share the exact watermark value are
  re-checked and de-duplicated, so late rows with the same timestamp aren't lost.
* **files** — a folder/glob: only files that are new or changed (size or mtime) are read.

State lives in a small JSON file and is committed only after you've processed the batch
(``with loader.batch() as df: …`` or ``loader.run(sink)``), so a crash mid-way re-delivers the
same rows instead of silently skipping them. Two runners can't commit over each other.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as _dt
import os
from pathlib import Path

import polars as pl

from ..registry._store import FileLock, now, read_json, write_json
from ._util import from_json_value, to_json_value


@dataclasses.dataclass
class Batch:
    data: pl.DataFrame
    new_state: dict
    base_version: int
    info: dict

    @property
    def empty(self) -> bool:
        return self.data.height == 0


class IncrementalLoader:
    """>>> loader = IncrementalLoader(SQLSource(url), table="orders", cursor="updated_at",
    ...                              key="order_id", state="state/orders.json")
    >>> with loader.batch() as new_rows:          # state is saved only if this block succeeds
    ...     append_to_warehouse(new_rows)

    >>> files = IncrementalLoader(files="landing/*.csv", state="state/landing.json")
    >>> files.run(lambda df: df.write_parquet(...))
    """

    def __init__(self, source=None, *, state, table: str | None = None, cursor: str | None = None,
                 key=None, files: str | None = None, read_kw: dict | None = None, infer: bool = True,
                 columns=None, batch_limit: int | None = None):
        if files is None and (source is None or cursor is None):
            raise ValueError("Pass files='glob', or a source plus cursor='column' (and table= for databases).")
        if files is None and not callable(source) and table is None:
            raise ValueError("Pass table='name' for a database source (or a callable source(watermark)).")
        self.source, self.table, self.cursor, self.files = source, table, cursor, files
        self.key = [key] if isinstance(key, str) else (list(key) if key else None)
        self.read_kw, self.infer, self.columns, self.batch_limit = dict(read_kw or {}), infer, columns, batch_limit
        self.state_path = Path(state)
        self._types_path = self.state_path.with_name(self.state_path.stem + ".types.json")

    # ── state ───────────────────────────────────────────────────────────────
    def state(self) -> dict:
        if self.state_path.exists():
            return read_json(self.state_path)
        return {"version": 0, "watermark": None, "boundary_keys": [], "files": {}, "history": []}

    @property
    def watermark(self):
        return from_json_value(self.state()["watermark"])

    def reset(self):
        """Forget everything: the next load is a full load."""
        self.state_path.unlink(missing_ok=True)
        self._types_path.unlink(missing_ok=True)

    # ── loading ─────────────────────────────────────────────────────────────
    def _read_cursor(self, watermark):
        inclusive = watermark is not None and self.key is not None
        if callable(self.source) and not hasattr(self.source, "read_table"):
            df = self.source(watermark)
            df = df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
            if watermark is not None and df.height:
                df = df.filter(pl.col(self.cursor) >= watermark if inclusive else pl.col(self.cursor) > watermark)
            return df.sort(self.cursor)
        return self.source.read_table(self.table, columns=self.columns, since=(self.cursor, watermark, inclusive),
                                      order_by=self.cursor, limit=self.batch_limit, **self.read_kw)

    def load(self) -> Batch:
        """Fetch the next batch without committing it (see :meth:`commit`)."""
        from .files import list_files, read_file
        state = self.state()
        info = {"time": now()}
        new_state = {**state, "history": list(state.get("history", []))[-49:]}
        if self.files is not None:
            seen = state.get("files", {})
            fresh, sigs = [], {}
            for p in list_files(self.files):
                st = os.stat(p)
                sig = [st.st_size, int(st.st_mtime_ns)]
                if seen.get(p) != sig:
                    fresh.append(p)
                sigs[p] = sig
            frames = [read_file(p, include_source=True, **self.read_kw) for p in fresh]
            df = pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
            new_state["files"] = {**seen, **{p: sigs[p] for p in fresh}}
            info["files"] = fresh
        else:
            watermark = from_json_value(state.get("watermark"))
            df = self._read_cursor(watermark)
            if df.height and watermark is not None and self.key and state.get("boundary_keys"):
                done = {tuple(k) for k in state["boundary_keys"]}
                at_edge = df[self.cursor] == watermark
                seen_before = pl.Series([tuple(r) in done for r in df.select(self.key).iter_rows()])
                df = df.filter(~(at_edge & seen_before))
            if self.key and df.height:
                df = df.unique(subset=self.key, keep="last", maintain_order=True)
            if df.height:
                top = df[self.cursor].max()
                new_state["watermark"] = to_json_value(top)
                keys = df.filter(pl.col(self.cursor) == top).select(self.key).rows() if self.key else []
                carry = state.get("boundary_keys", []) if from_json_value(state.get("watermark")) == top else []
                new_state["boundary_keys"] = [list(k) for k in {tuple(k) for k in carry} | set(keys)]
            info["watermark_from"] = state.get("watermark")
            info["watermark_to"] = new_state.get("watermark")
        if self.infer and df.height:
            from ..core import Pipeline
            from .schema import infer_types
            pipe = Pipeline.load(self._types_path) if self._types_path.exists() else None
            df, pipe, casts = infer_types(df, pipe)
            info["casts"] = casts
            self._pending_types = pipe
        else:
            self._pending_types = None
        info["rows"] = df.height
        new_state["history"].append({k: v for k, v in info.items() if k != "files"} |
                                    ({"files": len(info["files"])} if "files" in info else {}))
        return Batch(df, new_state, state.get("version", 0), info)

    def commit(self, batch: Batch):
        """Save the batch's watermark / file list. Fails if another run committed in between."""
        with FileLock(self.state_path.with_name(self.state_path.name + ".lock")):
            current = self.state().get("version", 0)
            if current != batch.base_version:
                raise RuntimeError(f"State {self.state_path} changed since this batch was loaded "
                                   "(another loader committed); reload and try again.")
            if self._pending_types is not None and not self._types_path.exists():
                self._pending_types.save(self._types_path)
            write_json(self.state_path, {**batch.new_state, "version": current + 1, "committed_at": now()})

    @contextlib.contextmanager
    def batch(self):
        """``with loader.batch() as df:`` — commits only if the block finishes without error."""
        b = self.load()
        yield b.data
        self.commit(b)

    def run(self, sink=None) -> dict:
        """Load, hand the rows to ``sink(df)`` (if any rows), commit; return the batch info."""
        b = self.load()
        if sink is not None and not b.empty:
            sink(b.data)
        self.commit(b)
        return b.info

    def __repr__(self) -> str:
        what = f"files={self.files!r}" if self.files else f"table={self.table!r}, cursor={self.cursor!r}"
        return f"IncrementalLoader({what}, state={str(self.state_path)!r})"
