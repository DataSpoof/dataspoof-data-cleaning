"""Target-leakage detection.

Leakage is information in the features that wouldn't exist at prediction time, typically a
column recorded *after* the outcome (``refund_issued`` when predicting churn) or a copy of the
target. It makes models look excellent in testing and fail in production.

``detect_leakage(df, target)`` scores every column **on its own** with cross-validation (a
small decision tree for numbers/dates, an out-of-fold category average for text). A single
column that predicts the target almost perfectly (default ≥ 0.98 ROC-AUC / balanced accuracy /
R²) is flagged ``high``. Columns whose name contains the target's name are flagged ``medium``.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections import Counter, defaultdict

import numpy as np
import polars as pl

from .._detect import is_num, is_temporal, tokens
from ..core import Step, register_step
from .selection import target_vector


class LeakageError(Exception):
    """Raised by ``DropLeakage(on_detect="raise")``; ``.report`` holds the LeakageReport."""

    def __init__(self, report: "LeakageReport"):
        self.report = report
        super().__init__(report.summary())


@dataclasses.dataclass
class LeakageFinding:
    column: str
    severity: str          # high | medium
    reason: str
    score: float | None
    detail: str


@dataclasses.dataclass
class LeakageReport:
    target: str
    task: str
    metric: str
    scores: dict           # column -> single-feature cross-validated score
    findings: list

    @property
    def leaky_columns(self) -> list:
        """Columns flagged ``high``: almost certainly leaking."""
        return [f.column for f in self.findings if f.severity == "high"]

    def to_dict(self) -> dict:
        return {"target": self.target, "task": self.task, "metric": self.metric,
                "scores": self.scores, "findings": [dataclasses.asdict(f) for f in self.findings]}

    def to_frame(self) -> pl.DataFrame:
        schema = {"column": pl.String, "severity": pl.String, "reason": pl.String,
                  "score": pl.Float64, "detail": pl.String}
        return pl.DataFrame([dataclasses.asdict(f) for f in self.findings], schema=schema)

    def summary(self) -> str:
        if not self.findings:
            return f"No target leakage detected for '{self.target}' ({len(self.scores)} columns checked)."
        lines = [f"Possible target leakage for '{self.target}' ({self.task}, metric {self.metric}):"]
        lines += [f"  [{f.severity}] {f.column}: {f.detail}" for f in self.findings]
        return "\n".join(lines)

    __str__ = summary

    def __repr__(self) -> str:
        return self.summary()


def _numeric_vector(s: pl.Series) -> np.ndarray:
    if s.dtype == pl.Boolean or is_temporal(s.dtype):
        s = s.cast(pl.Int64)
    x = s.cast(pl.Float64).to_numpy().astype(float)
    if np.isnan(x).all():
        return np.zeros_like(x)
    return np.where(np.isnan(x), np.nanmin(x) - 1.0, x)   # nulls get their own region


def _single_feature_score(x, y, kind: str, categorical: bool, cv: int, seed: int) -> float | None:
    from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    classify = kind != "regression"
    if classify:
        _, counts = np.unique(y, return_counts=True)
        folds = min(cv, int(counts.min()))
        if folds < 2:
            return None
        splitter = StratifiedKFold(folds, shuffle=True, random_state=seed)
    else:
        splitter = KFold(cv, shuffle=True, random_state=seed)
    preds = np.zeros(len(y), dtype=float)
    for tr, te in splitter.split(np.zeros(len(y)), y if classify else None):
        if categorical:      # out-of-fold category average (or majority class), one pass
            if kind == "multiclass":
                tallies = {}
                for v, t in zip(x[tr], y[tr]):
                    tallies.setdefault(v, Counter())[t] += 1
                prior = Counter(y[tr].tolist()).most_common(1)[0][0]
                table = {k: c.most_common(1)[0][0] for k, c in tallies.items()}
            else:
                sums, counts = defaultdict(float), defaultdict(int)
                for v, t in zip(x[tr], y[tr]):
                    sums[v] += t
                    counts[v] += 1
                prior = float(np.mean(y[tr]))
                table = {k: sums[k] / counts[k] for k in sums}
            preds[te] = [table.get(v, prior) for v in x[te]]
        else:
            model = (DecisionTreeClassifier if classify else DecisionTreeRegressor)(
                max_depth=4, random_state=seed).fit(x[tr].reshape(-1, 1), y[tr])
            if kind == "binary":
                proba = model.predict_proba(x[te].reshape(-1, 1))
                preds[te] = proba[:, list(model.classes_).index(1)] if 1 in model.classes_ else 0.0
            else:
                preds[te] = model.predict(x[te].reshape(-1, 1))
    if kind == "binary":
        return float(roc_auc_score(y, preds)) if len(set(y)) == 2 else None
    if kind == "multiclass":
        return float(balanced_accuracy_score(y, preds.astype(int)))
    return float(r2_score(y, preds))


def detect_leakage(df, target: str, *, threshold: float = 0.98, cv: int = 3, sample: int = 5000,
                   seed: int = 42) -> LeakageReport:
    """Score each column alone against ``target`` and report suspiciously strong ones."""
    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)
    if target not in df.columns:
        raise ValueError(f"Target '{target}' is not in the data")
    df = df.filter(pl.col(target).is_not_null())
    if df.height > sample:
        df = df.sample(sample, seed=seed)
    kind, y = target_vector(df[target])
    metric = {"binary": "roc_auc", "multiclass": "balanced_accuracy", "regression": "r2"}[kind]
    target_tokens = tokens(target)
    scores, findings = {}, []
    for c, dtype in df.schema.items():
        if c == target:
            continue
        s = df[c]
        if s.drop_nulls().n_unique() <= 1:
            continue
        categorical = not (is_num(dtype) or dtype == pl.Boolean or is_temporal(dtype))
        x = (np.array(s.cast(pl.String).fill_null("__null__").to_list(), dtype=object)
             if categorical else _numeric_vector(s))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score = _single_feature_score(x, y, kind, categorical, cv, seed)
        except Exception:
            score = None
        if score is None:
            continue
        scores[c] = round(score, 4)
        if score >= threshold:
            findings.append(LeakageFinding(
                c, "high", "predicts target alone", round(score, 4),
                f"on its own it predicts '{target}' with cross-validated {metric} = {score:.3f}; "
                f"check it isn't recorded after the outcome or derived from the target."))
        elif target_tokens and target_tokens <= tokens(c):
            findings.append(LeakageFinding(
                c, "medium", "name mentions target", round(score, 4),
                f"its name contains '{target}' — make sure it's known before the outcome "
                f"(single-feature {metric} = {score:.3f})."))
    findings.sort(key=lambda f: (f.severity != "high", -(f.score or 0)))
    return LeakageReport(target=target, task=kind, metric=metric, scores=scores, findings=findings)


@register_step
class DropLeakage(Step):
    """Detect target leakage while fitting and drop (or warn about / refuse) leaky columns.

    ``on_detect``: ``"drop"`` (default) removes ``high`` findings (plus ``medium`` ones if
    ``drop_suspicious_names``); ``"warn"`` keeps them but warns; ``"raise"`` stops with a
    ``LeakageError``. The report is kept in ``report_`` and the findings in the saved state.
    """

    name = "drop_leakage"
    uses_target = True

    def __init__(self, *, target=None, threshold=0.98, drop_suspicious_names=False,
                 on_detect="drop", cv=3, sample=5000, seed=42, fit_only=None):
        if on_detect not in ("drop", "warn", "raise"):
            raise ValueError("on_detect must be 'drop', 'warn' or 'raise'")
        super().__init__(fit_only=fit_only, target=target, threshold=threshold,
                         drop_suspicious_names=drop_suspicious_names, on_detect=on_detect,
                         cv=cv, sample=sample, seed=seed)
        self.report_ = None

    def _fit(self, df):
        p = self.params
        target = self._target()
        report = detect_leakage(df, target, threshold=p["threshold"], cv=p["cv"],
                                sample=p["sample"], seed=p["seed"])
        self.report_ = report
        flagged = [f.column for f in report.findings
                   if f.severity == "high" or (p["drop_suspicious_names"] and f.severity == "medium")]
        if flagged and p["on_detect"] == "raise":
            raise LeakageError(report)
        if flagged and p["on_detect"] == "warn":
            with warnings.catch_warnings():
                warnings.simplefilter("always", UserWarning)
                warnings.warn(report.summary(), UserWarning, stacklevel=5)
        return {"target": target, "drop": flagged if p["on_detect"] == "drop" else [],
                "findings": [dataclasses.asdict(f) for f in report.findings]}

    def _transform(self, df):
        drop = [c for c in self.state_["drop"] if c in df.columns]
        return df.drop(drop) if drop else df
