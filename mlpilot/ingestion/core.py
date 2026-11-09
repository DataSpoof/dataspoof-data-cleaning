"""One entry point for every source: ``read()`` returns a DataFrame, ``ingest()`` adds types,
validation, a data-contract check, lineage and an audit record."""

from __future__ import annotations

from typing import Any

import dataclasses
import time

import polars as pl

from ..quality.checks import jsonable
from ._util import redact
from .files import detect_format, is_cloud, is_http, read_file
from .sql import SQLSource, is_sql_url


def read(source, *, query: str | None = None, table: str | None = None, params: dict | None = None, **kw):
    """Read from anything:

    * a path, glob, ``http(s)://`` file URL or cloud URI (``s3://``, ``gs://``, ``az://``, ``abfss://``)
    * a database URL (``postgresql://…``, ``mysql+pymysql://…``, ``sqlite:///file.db`` …) with
      ``query=`` (``:name`` params in ``params``) or ``table=``
    * a source object (``SQLSource``, ``SnowflakeSource``, ``BigQuerySource``, ``RESTSource`` …)
    * an existing Polars / pandas DataFrame (passed through)
    """
    if isinstance(source, pl.DataFrame):
        return source
    if isinstance(source, pl.LazyFrame):
        return source.collect()
    if type(source).__module__.startswith("pandas"):
        return pl.from_pandas(source)
    if hasattr(source, "read") and not isinstance(source, (str, bytes)):
        return source.read(query=query, table=table, params=params, **kw) if (query or table) else source.read(**kw)
    text = str(source)
    if is_sql_url(text):
        db = SQLSource(text)
        try:
            return db.read(query=query, table=table, params=params, **kw)
        finally:
            db.close()
    if is_http(text) and detect_format(text) is None:
        from .api import RESTSource
        return RESTSource(text).fetch(**kw)
    return read_file(text, **kw)


@dataclasses.dataclass
class IngestResult:
    """The loaded data plus everything you'd want in an audit log."""

    data: pl.DataFrame
    source: str
    name: str | None
    rows: int
    columns: int
    seconds: float
    schema: dict
    casts: dict
    fingerprint: dict
    contract: Any = None
    quality: dict | None = None
    types_pipeline: Any = None

    @property
    def passed(self) -> bool:
        return self.contract is None or self.contract.passed

    def summary(self) -> str:
        lines = [f"Ingested {self.name or 'data'} from {self.source}: {self.rows:,} rows x {self.columns} columns "
                 f"in {self.seconds:.2f}s"]
        if self.casts:
            lines.append("  Types inferred: " + ", ".join(f"{c} ({v.split(' -> ')[-1]})" for c, v in self.casts.items()))
        if self.contract is not None:
            lines.append(f"  Contract: {'passed' if self.contract.passed else 'FAILED'} "
                         f"({len(self.contract.errors)} error(s), {len(self.contract.warnings)} warning(s))")
        if self.quality:
            lines.append(f"  Quality: {self.quality['score']:.0f}/100 (grade {self.quality['grade']})")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        out = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)
               if f.name not in ("data", "types_pipeline")}
        out["contract"] = self.contract.to_dict() if self.contract is not None else None
        return jsonable(out)

    def __repr__(self) -> str:
        return self.summary()


def ingest(source, *, name: str | None = None, infer: bool = True, types=None, contract=None,
           on_violation: str = "raise", quality: bool = False, run=None, **read_kw) -> IngestResult:
    """Read, infer types, validate and record a dataset.

    * ``infer``: convert text columns holding numbers / dates / yes-no to real types; pass a
      fitted ``types`` pipeline (``result.types_pipeline``) to reuse the same conversions.
    * ``contract``: a ``Contract`` (or path to its YAML) to validate against; ``on_violation``
      ``"raise"`` (``ContractViolation``), ``"warn"`` or ``"ignore"``.
    * ``quality=True`` adds the 0-100 quality score.
    * ``run``: a tracking ``Run`` — the dataset fingerprint is logged to it.
    """
    from ..registry._store import dataset_fingerprint
    t0 = time.perf_counter()
    df = read(source, **read_kw)
    casts, pipe = {}, None
    if infer and df.height:
        from .schema import infer_types
        df, pipe, casts = infer_types(df, types)
    label = redact(source) if isinstance(source, str) else repr(source) if not isinstance(
        source, pl.DataFrame) else "DataFrame"
    result = IngestResult(data=df, source=label, name=name, rows=df.height, columns=df.width,
                          seconds=round(time.perf_counter() - t0, 3),
                          schema={c: str(t) for c, t in df.schema.items()}, casts=casts,
                          fingerprint=dataset_fingerprint(df, name), types_pipeline=pipe)
    if contract is not None:
        from ..quality import Contract
        if not isinstance(contract, Contract):
            contract = Contract.from_file(contract)
        result.contract = contract.validate(df)
        if not result.contract.passed:
            if on_violation == "raise":
                result.contract.raise_if_failed()
            elif on_violation == "warn":
                import warnings

                from ..core import MLPilotWarning
                warnings.warn(f"{name or 'data'} failed its contract: {len(result.contract.errors)} error(s)",
                              MLPilotWarning, stacklevel=2)
    if quality:
        from ..quality import check_quality
        q = check_quality(df)
        result.quality = {"score": q.score, "grade": q.grade, "issues": len(q.issues)}
    if run is not None:
        run.log_dataset(df, name or "ingested")
    return result
