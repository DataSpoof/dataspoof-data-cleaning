"""Detailed evaluation of a fitted model on labelled data."""

from __future__ import annotations

import dataclasses
import json
import warnings

import numpy as np
import polars as pl

from ..quality.checks import jsonable
from .adapter import ModelAdapter

_CLASSIFICATION_HEADLINE = ("accuracy", "balanced_accuracy", "f1_macro", "roc_auc", "roc_auc_ovr")
_REGRESSION_HEADLINE = ("rmse", "mae", "r2")


def _r(v, digits=6):
    return None if v is None or (isinstance(v, float) and v != v) else round(float(v), digits)


@dataclasses.dataclass
class EvaluationReport:
    """Everything about how well a model does on one labelled dataset."""

    task: str
    model: str
    target: str
    n_rows: int
    metrics: dict
    confidence_intervals: dict = dataclasses.field(default_factory=dict)
    classes: list | None = None
    positive_class: object = None
    threshold: float | None = None
    confusion_matrix: pl.DataFrame | None = None
    per_class: pl.DataFrame | None = None
    calibration: pl.DataFrame | None = None
    roc_curve: pl.DataFrame | None = None
    pr_curve: pl.DataFrame | None = None
    thresholds: pl.DataFrame | None = None
    residuals: dict | None = None
    error_by_range: pl.DataFrame | None = None

    @property
    def is_classification(self) -> bool:
        return self.task != "regression"

    def best_threshold(self, by: str = "f1") -> float:
        """Decision threshold that maximises ``f1``, ``youden`` (TPR − FPR) or ``accuracy``."""
        if self.thresholds is None:
            raise ValueError("Threshold analysis is only available for binary classification.")
        if by == "youden":
            score = self.thresholds["recall"] + self.thresholds["specificity"] - 1
        elif by in ("f1", "accuracy", "precision", "recall"):
            score = self.thresholds[by]
        else:
            raise ValueError("by must be f1, youden, accuracy, precision or recall")
        return float(self.thresholds["threshold"][int(score.fill_null(-1).arg_max())])

    def summary(self) -> str:
        lines = [f"Evaluation of {self.model} on {self.n_rows:,} rows ({self.task} '{self.target}')"]
        for k, v in self.metrics.items():
            if v is None:
                continue
            ci = self.confidence_intervals.get(k)
            extra = f"  (95% CI {ci[0]:.4f} to {ci[1]:.4f})" if ci else ""
            lines.append(f"  {k:<22} {v:.4f}{extra}")
        if self.per_class is not None and len(self.per_class) <= 12:
            lines.append("  Per class:")
            for row in self.per_class.iter_rows(named=True):
                lines.append(f"    {str(row['class']):<16} precision {row['precision']:.3f}  recall "
                             f"{row['recall']:.3f}  f1 {row['f1']:.3f}  n={row['support']}")
        if self.thresholds is not None:
            lines.append(f"  Best F1 threshold: {self.best_threshold('f1'):.2f}"
                         f" (current {self.threshold if self.threshold is not None else 0.5:.2f})")
        if self.residuals:
            r = self.residuals
            lines.append(f"  Residuals: bias {r['mean']:+.4g}, std {r['std']:.4g}; 90% of errors within "
                         f"[{r['p05']:+.4g}, {r['p95']:+.4g}]")
        return "\n".join(lines)

    def to_dict(self, curves: bool = True) -> dict:
        out = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if isinstance(v, pl.DataFrame):
                if not curves and f.name in ("roc_curve", "pr_curve"):
                    continue
                v = v.to_dicts()
            out[f.name] = v
        return jsonable(out)

    def to_json(self, path=None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent, default=str)
        if path is not None:
            from pathlib import Path
            Path(path).write_text(text, encoding="utf-8")
        return text

    def __repr__(self) -> str:
        return self.summary()


# ── helpers ─────────────────────────────────────────────────────────────────

def _downsample(frame: pl.DataFrame, n: int = 200) -> pl.DataFrame:
    if frame.height <= n:
        return frame
    idx = np.unique(np.linspace(0, frame.height - 1, n).round().astype(int))
    return frame[idx.tolist()]


def reliability(confidence: np.ndarray, correct: np.ndarray, bins: int = 10):
    """Calibration table (equal-width bins) and the expected calibration error."""
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(confidence, edges[1:-1]), 0, bins - 1)
    rows, ece, n = [], 0.0, len(confidence)
    for b in range(bins):
        mask = which == b
        if not mask.any():
            continue
        mean_p, observed = float(confidence[mask].mean()), float(correct[mask].mean())
        ece += mask.sum() / n * abs(mean_p - observed)
        rows.append({"bin": f"{edges[b]:.1f}–{edges[b + 1]:.1f}", "count": int(mask.sum()),
                     "mean_predicted": round(mean_p, 4), "observed": round(observed, 4),
                     "gap": round(observed - mean_p, 4)})
    return pl.DataFrame(rows), float(ece)


def _bootstrap(fn, arrays, n_boot: int, seed: int, stratify=None):
    """95% percentile interval of ``fn(*arrays)`` over bootstrap resamples."""
    if n_boot <= 0:
        return None
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if stratify is not None and len(np.unique(stratify[idx])) < len(np.unique(stratify)):
            continue                      # e.g. a resample with one class has no ROC-AUC
        try:
            v = fn(*[a[idx] for a in arrays])
        except ValueError:
            continue
        if v is not None and v == v:
            stats.append(v)
    if len(stats) < max(20, n_boot // 4):
        return None
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return [round(float(lo), 6), round(float(hi), 6)]


# ── classification ──────────────────────────────────────────────────────────

def _classification(ad: ModelAdapter, y, pred, proba, report: dict, threshold, pos, n_boot, seed):
    from sklearn import metrics as m

    k = len(ad.classes)
    labels = list(range(k))
    names = [str(c) for c in ad.classes]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = {"accuracy": m.accuracy_score(y, pred),
               "balanced_accuracy": m.balanced_accuracy_score(y, pred),
               "precision_macro": m.precision_score(y, pred, average="macro", labels=labels, zero_division=0),
               "recall_macro": m.recall_score(y, pred, average="macro", labels=labels, zero_division=0),
               "f1_macro": m.f1_score(y, pred, average="macro", labels=labels, zero_division=0),
               "f1_weighted": m.f1_score(y, pred, average="weighted", labels=labels, zero_division=0),
               "mcc": m.matthews_corrcoef(y, pred),
               "cohen_kappa": m.cohen_kappa_score(y, pred, labels=labels)}
        both = len(np.unique(y)) == k
        if k == 2:
            p, yb, pb = proba[:, pos], (y == pos).astype(int), (pred == pos).astype(int)
            tn, fp, fn, tp = m.confusion_matrix(yb, pb, labels=[0, 1]).ravel()
            out.update({"precision": m.precision_score(yb, pb, zero_division=0),
                        "recall": m.recall_score(yb, pb, zero_division=0),
                        "f1": m.f1_score(yb, pb, zero_division=0),
                        "specificity": tn / (tn + fp) if tn + fp else None,
                        "roc_auc": m.roc_auc_score(yb, p) if both else None,
                        "pr_auc": m.average_precision_score(yb, p) if both else None,
                        "brier": m.brier_score_loss(yb, p),
                        "log_loss": m.log_loss(y, proba, labels=labels)})
            confidence, correct = p, yb
        else:
            onehot = np.eye(k)[y]
            out.update({"roc_auc_ovr": m.roc_auc_score(y, proba, multi_class="ovr", labels=labels)
                        if both else None,
                        "brier": float(np.mean(np.sum((proba - onehot) ** 2, axis=1))),
                        "log_loss": m.log_loss(y, proba, labels=labels),
                        "top2_accuracy": m.top_k_accuracy_score(y, proba, k=2, labels=labels)
                        if k > 2 else None})
            confidence, correct = proba.max(axis=1), (proba.argmax(axis=1) == y).astype(int)
    report["calibration"], out["ece"] = reliability(confidence, correct)
    report["metrics"] = {k_: _r(v) for k_, v in out.items()}

    cm = m.confusion_matrix(y, pred, labels=labels)
    report["confusion_matrix"] = pl.DataFrame(
        {"actual": names, **{f"pred_{n}": cm[:, j].tolist() for j, n in enumerate(names)}})
    prec, rec, f1, support = m.precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    report["per_class"] = pl.DataFrame({
        "class": names, "precision": np.round(prec, 4), "recall": np.round(rec, 4),
        "f1": np.round(f1, 4), "support": support.astype(int), "predicted": cm.sum(axis=0).astype(int)})

    ci = {"accuracy": _bootstrap(lambda a, b: m.accuracy_score(a, b), [y, pred], n_boot, seed),
          "f1_macro": _bootstrap(lambda a, b: m.f1_score(a, b, average="macro", labels=labels,
                                                         zero_division=0), [y, pred], n_boot, seed)}
    if k == 2 and out["roc_auc"] is not None:
        ci["roc_auc"] = _bootstrap(lambda a, s: m.roc_auc_score(a, s), [yb, p], n_boot, seed, stratify=yb)
    report["confidence_intervals"] = {k_: v for k_, v in ci.items() if v}

    if k == 2 and both:
        fpr, tpr, thr = m.roc_curve(yb, p)
        report["roc_curve"] = _downsample(pl.DataFrame({"fpr": fpr, "tpr": tpr,
                                                        "threshold": np.clip(thr, 0, 1)}))
        precision, recall, thr = m.precision_recall_curve(yb, p)
        report["pr_curve"] = _downsample(pl.DataFrame({"recall": recall[:-1], "precision": precision[:-1],
                                                       "threshold": thr}))
        rows = []
        for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
            pt = (p >= t).astype(int)
            tn, fp, fn, tp = m.confusion_matrix(yb, pt, labels=[0, 1]).ravel()
            rows.append({"threshold": float(t),
                         "precision": _r(tp / (tp + fp), 4) if tp + fp else None,
                         "recall": _r(tp / (tp + fn), 4) if tp + fn else None,
                         "specificity": _r(tn / (tn + fp), 4) if tn + fp else None,
                         "f1": _r(2 * tp / (2 * tp + fp + fn), 4) if tp + fp + fn else None,
                         "accuracy": _r((tp + tn) / len(pt), 4),
                         "predicted_positive_rate": _r(pt.mean(), 4)})
        report["thresholds"] = pl.DataFrame(rows, infer_schema_length=None)


# ── regression ──────────────────────────────────────────────────────────────

def _regression(y, pred, report: dict, n_boot, seed, bins: int = 5):
    from sklearn import metrics as m

    res = pred - y
    out = {"rmse": float(np.sqrt(m.mean_squared_error(y, pred))), "mae": m.mean_absolute_error(y, pred),
           "median_ae": m.median_absolute_error(y, pred), "max_error": m.max_error(y, pred),
           "r2": m.r2_score(y, pred), "explained_variance": m.explained_variance_score(y, pred),
           "bias": float(res.mean())}
    if not np.any(y == 0):
        out["mape"] = m.mean_absolute_percentage_error(y, pred)
    denom = np.abs(y) + np.abs(pred)
    out["smape"] = float(np.mean(np.where(denom == 0, 0.0, 2 * np.abs(res) / np.where(denom == 0, 1, denom))))
    report["metrics"] = {k: _r(v) for k, v in out.items()}
    std = res.std()
    report["residuals"] = {"mean": _r(res.mean()), "std": _r(std),
                           "skew": _r(((res - res.mean()) ** 3).mean() / std ** 3) if std > 0 else 0.0,
                           "p05": _r(np.percentile(res, 5)), "p50": _r(np.percentile(res, 50)),
                           "p95": _r(np.percentile(res, 95))}
    report["confidence_intervals"] = {k: v for k, v in {
        "rmse": _bootstrap(lambda a, b: float(np.sqrt(np.mean((a - b) ** 2))), [y, pred], n_boot, seed),
        "mae": _bootstrap(lambda a, b: float(np.mean(np.abs(a - b))), [y, pred], n_boot, seed),
        "r2": _bootstrap(lambda a, b: m.r2_score(a, b), [y, pred], n_boot, seed),
    }.items() if v}

    edges = np.unique(np.quantile(y, np.linspace(0, 1, bins + 1)))
    if len(edges) >= 2:
        which = np.clip(np.digitize(y, edges[1:-1], right=False), 0, len(edges) - 2)
        rows = []
        for b in range(len(edges) - 1):
            mask = which == b
            if mask.any():
                rows.append({"range": f"[{edges[b]:.4g}, {edges[b + 1]:.4g}{']' if b == len(edges) - 2 else ')'}",
                             "count": int(mask.sum()), "mean_actual": _r(y[mask].mean(), 4),
                             "mean_predicted": _r(pred[mask].mean(), 4),
                             "mae": _r(np.abs(res[mask]).mean(), 4), "bias": _r(res[mask].mean(), 4)})
        report["error_by_range"] = pl.DataFrame(rows)


# ── entry point ─────────────────────────────────────────────────────────────

def evaluate_model(model, df, *, target: str | None = None, threshold: float | None = None,
                   pos_label=None, bootstrap: int = 200, seed: int = 42) -> EvaluationReport:
    """Detailed metrics for a fitted ``AutoML`` model (or scikit-learn estimator + ``target``).

    Classification: accuracy, balanced accuracy, precision/recall/F1 (macro, weighted and
    per class), MCC, Cohen's kappa, ROC-AUC, PR-AUC, log loss, Brier score, a confusion matrix,
    a calibration table with the expected calibration error (ECE), and — for binary models —
    ROC/PR curves and a threshold table. ``threshold`` changes the binary decision cut-off.

    Regression: RMSE, MAE, median AE, max error, R², explained variance, bias, MAPE/sMAPE,
    residual distribution and error by range of the actual value.

    Headline metrics get 95% bootstrap confidence intervals (``bootstrap`` resamples; 0 = off).
    """
    ad = ModelAdapter.of(model, target)
    data = ad.labelled(df)
    if data.height == 0:
        raise ValueError("No labelled rows to evaluate.")
    X, y = ad.X(data), ad.y(data)
    report = {"task": ad.task, "model": ad.name, "target": ad.target, "n_rows": data.height,
              "classes": ad.classes}
    if ad.is_classification:
        proba = ad.proba(X)
        pos = ad.positive_index(pos_label) if ad.task == "binary" else None
        if pos is not None and threshold is not None:
            pred = np.where(proba[:, pos] >= threshold, pos, 1 - pos)
        else:
            if threshold is not None:
                raise ValueError("threshold only applies to binary classification.")
            pred = ad.predict(X)
        report.update(positive_class=ad.classes[pos] if pos is not None else None, threshold=threshold)
        _classification(ad, y, pred, proba, report, threshold, pos, bootstrap, seed)
    else:
        _regression(y, ad.predict(X), report, bootstrap, seed)
    return EvaluationReport(**report)
