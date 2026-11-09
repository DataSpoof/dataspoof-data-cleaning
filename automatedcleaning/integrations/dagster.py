"""Dagster software-defined assets (needs ``dagster``).

>>> from automatedcleaning.integrations.dagster import ml_assets
>>> defs = dagster.Definitions(assets=ml_assets(source="data/churn.parquet", target="churned",
...                                             registry="/models", name="churn"))
"""

from __future__ import annotations

try:
    import dagster as dg
except ImportError as exc:                                   # pragma: no cover
    raise ImportError("These assets need Dagster: pip install dagster") from exc

from . import tasks


def ml_assets(*, source, target: str, registry: str, name: str, score_source=None, score_output=None,
              promote_to: str = "production", require: dict | None = None, group: str = "ml"):
    """A ``trained_model`` asset and, with ``score_source``/``score_output``, a ``scored_batch``
    asset downstream of it."""
    @dg.asset(name=f"{name}_trained_model", group_name=group)
    def trained_model():
        out = tasks.train_and_register(source, target, registry=registry, name=name, promote_to=promote_to,
                                       require=require)
        return dg.Output(out, metadata={"version": out["version"], "promoted": str(out["promoted"])})

    assets = [trained_model]
    if score_source is not None and score_output is not None:
        @dg.asset(name=f"{name}_scored_batch", group_name=group, deps=[trained_model])
        def scored_batch():
            out = tasks.batch_score_from_registry(registry, name, score_source, score_output, stage=promote_to)
            return dg.Output(out, metadata={"rows": out["rows"], "version": out["version"]})
        assets.append(scored_batch)
    return assets
