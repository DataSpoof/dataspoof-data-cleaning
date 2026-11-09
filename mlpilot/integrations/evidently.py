"""Evidently integration (needs ``evidently>=0.7``; the ``[evidently]`` extra).

Use Evidently's reports and UI next to the built-in monitoring:

>>> from mlpilot.integrations.evidently import evidently_report
>>> res = evidently_report(reference_df, current_df, model=model, html="drift.html")
>>> res.drifted_columns, res.drift_share        # parsed from Evidently's own drift tests
>>> res.compare(ac.drift_report(reference_df, current_df))   # agreement with our PSI report
>>> res.snapshot                                # the Evidently Snapshot (save_json, dict, upload to a workspace)

With ``model=`` the model's predictions (and class probabilities) are added to both datasets,
so Evidently's classification / regression presets run as well when the target is present.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import polars as pl

try:
    from evidently import BinaryClassification, DataDefinition, Dataset, MulticlassClassification, Regression, Report
    from evidently.presets import ClassificationPreset, DataDriftPreset, RegressionPreset
except ImportError as exc:                                   # pragma: no cover
    raise ImportError('The Evidently integration needs evidently>=0.7: pip install "mlpilot[evidently]"') from exc


@dataclasses.dataclass
class EvidentlyResult:
    snapshot: Any
    column_drift: dict           # column -> {"method", "value", "threshold", "drift"}
    metrics: dict                # performance metrics Evidently computed (when labels were given)

    @property
    def drifted_columns(self) -> list:
        return sorted(c for c, d in self.column_drift.items() if d["drift"])

    @property
    def drift_share(self) -> float:
        return len(self.drifted_columns) / max(len(self.column_drift), 1)

    def compare(self, ours) -> dict:
        """Agreement with a :class:`~mlpilot.monitoring.DriftReport` (moderate or major
        drift counts as drift on our side)."""
        mine = set(ours.drifted)
        theirs = set(self.drifted_columns)
        common = set(self.column_drift) & set(ours.columns["column"].to_list())
        agree = sum(1 for c in common if (c in mine) == (c in theirs))
        return {"columns": len(common), "agreement": agree / max(len(common), 1), "both": sorted(mine & theirs),
                "only_ours": sorted((mine - theirs) & common), "only_evidently": sorted((theirs - mine) & common)}

    def save_html(self, path) -> None:
        self.snapshot.save_html(str(path))


def _with_predictions(df: pl.DataFrame, model) -> tuple:
    labels, proba = model.predict_with_proba(df)
    out = df.with_columns(labels.alias("prediction"))
    proba_cols = []
    if proba is not None:
        proba = proba.rename({c: c.replace("proba_", "") for c in proba.columns})
        proba_cols = proba.columns
        out = pl.concat([out, proba], how="horizontal")
    return out, proba_cols


def evidently_report(reference, current, *, model=None, target: str | None = None, columns: list | None = None,
                     html=None, json_path=None) -> EvidentlyResult:
    """Run Evidently's data-drift preset (plus classification / regression presets when a model
    and labels are available) on ``current`` vs ``reference``."""
    ref = reference if isinstance(reference, pl.DataFrame) else pl.from_pandas(reference)
    cur = current if isinstance(current, pl.DataFrame) else pl.from_pandas(current)
    target = target or getattr(model, "target", None)
    feats = columns or [c for c in ref.columns if c in cur.columns and c != target]
    keep = feats + ([target] if target and target in ref.columns and target in cur.columns else [])
    ref, cur = ref.select(keep), cur.select(keep)
    num = [c for c in feats if ref.schema[c].is_numeric()]
    cat = [c for c in feats if c not in num]
    presets: list = [DataDriftPreset(columns=feats)]
    task_defs: dict = {}
    if model is not None:
        ref, proba_cols = _with_predictions(ref, model)
        cur, _ = _with_predictions(cur, model)
        has_labels = target in ref.columns and target in cur.columns
        classes = getattr(model, "classes_", None)
        if classes is not None:
            for df_name in ("ref", "cur"):
                df = ref if df_name == "ref" else cur
                cast = [pl.col("prediction").cast(pl.String)] + ([pl.col(str(target)).cast(pl.String)] if has_labels else [])
                if df_name == "ref":
                    ref = df.with_columns(cast)
                else:
                    cur = df.with_columns(cast)
            if has_labels:
                if len(classes) == 2:
                    pos = str(classes[1])
                    task_defs["classification"] = [BinaryClassification(target=target, prediction_labels="prediction",
                                                                        prediction_probas=pos, pos_label=pos)]
                else:
                    task_defs["classification"] = [MulticlassClassification(target=target,
                                                                            prediction_labels="prediction",
                                                                            prediction_probas=proba_cols)]
                presets.append(ClassificationPreset())
        elif has_labels:
            task_defs["regression"] = [Regression(target=target, prediction="prediction")]
            presets.append(RegressionPreset())
        extra_cat = ["prediction"] + ([target] if classes is not None and has_labels else [])
        definition = DataDefinition(numerical_columns=num + ([] if classes is not None else ["prediction"]
                                                              + ([target] if has_labels else [])) + proba_cols,
                                    categorical_columns=cat + (extra_cat if classes is not None else []),
                                    **task_defs)
    else:
        definition = DataDefinition(numerical_columns=num, categorical_columns=cat)
    ds_ref = Dataset.from_pandas(ref.to_pandas(), data_definition=definition)
    ds_cur = Dataset.from_pandas(cur.to_pandas(), data_definition=definition)
    snap = Report(presets).run(ds_cur, ds_ref)
    drift: dict = {}
    metrics: dict = {}
    for m in snap.dict()["metrics"]:
        cfg = m.get("config", {})
        kind = str(cfg.get("type", ""))
        if kind.endswith("ValueDrift") and cfg.get("column") in feats:
            method, thr, val = str(cfg.get("method", "")), float(cfg.get("threshold", 0.05)), float(m["value"])
            is_p = "p_value" in method
            drift[cfg["column"]] = {"method": method, "value": val, "threshold": thr,
                                    "drift": val < thr if is_p else val >= thr}
        elif not isinstance(m.get("value"), dict) or kind.endswith(("Accuracy", "F1Score", "RocAuc", "MAE", "RMSE",
                                                                     "Precision", "Recall", "R2Score")):
            name = kind.split(":")[-1]
            val = m.get("value")
            if isinstance(val, dict) and "mean" in val:
                val = val["mean"]
            if isinstance(val, (int, float)) and name not in ("ValueDrift", "DriftedColumnsCount"):
                metrics[name] = float(val)
    if html:
        snap.save_html(str(html))
    if json_path:
        snap.save_json(str(json_path))
    return EvidentlyResult(snapshot=snap, column_drift=drift, metrics=metrics)
