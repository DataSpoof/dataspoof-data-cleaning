"""Schema inference and comparison: catch added, removed and retyped columns."""

from __future__ import annotations

import dataclasses

import polars as pl


def infer_schema(df) -> dict:
    """``{column: dtype-string}`` for a DataFrame (pandas is converted)."""
    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)
    return {c: str(t) for c, t in df.schema.items()}


@dataclasses.dataclass
class SchemaDiff:
    """Differences between an expected and an actual schema."""

    added: list
    removed: list
    changed: dict          # column -> [expected dtype, actual dtype]

    @property
    def identical(self) -> bool:
        return not (self.added or self.removed or self.changed)

    @property
    def compatible(self) -> bool:
        """True when nothing expected is missing or retyped (new columns are tolerated)."""
        return not (self.removed or self.changed)

    def to_dict(self) -> dict:
        return {**dataclasses.asdict(self), "identical": self.identical,
                "compatible": self.compatible}

    def summary(self) -> str:
        if self.identical:
            return "Schemas are identical."
        lines = [f"Schema {'compatible' if self.compatible else 'INCOMPATIBLE'}:"]
        if self.removed:
            lines.append(f"  removed: {self.removed}")
        if self.changed:
            lines += [f"  type changed: {c} {old} -> {new}" for c, (old, new) in self.changed.items()]
        if self.added:
            lines.append(f"  added: {self.added}")
        return "\n".join(lines)

    __str__ = summary

    def __repr__(self) -> str:
        return self.summary()


def compare_schemas(expected, actual) -> SchemaDiff:
    """Compare two schemas; each may be a DataFrame or a ``{column: dtype}`` dict."""
    exp = expected if isinstance(expected, dict) else infer_schema(expected)
    act = actual if isinstance(actual, dict) else infer_schema(actual)
    exp, act = {k: str(v) for k, v in exp.items()}, {k: str(v) for k, v in act.items()}
    return SchemaDiff(
        added=[c for c in act if c not in exp],
        removed=[c for c in exp if c not in act],
        changed={c: [exp[c], act[c]] for c in exp if c in act and exp[c] != act[c]},
    )
