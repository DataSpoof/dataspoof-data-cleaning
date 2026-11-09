"""Auto-generated model cards (Mitchell et al., 2019) in Markdown, HTML and JSON."""

from __future__ import annotations

from typing import Any

import dataclasses
import datetime as _dt
import html
import json
import platform
import re
from pathlib import Path

import numpy as np

from ..quality.checks import jsonable
from .adapter import ModelAdapter, as_polars

SENSITIVE_HINTS = re.compile(
    r"(^|_)(sex|gender|race|ethnic\w*|age|religion|nationality|citizenship|marital|disab\w*|"
    r"pregnan\w*|caste|orientation|color|colour)($|_)", re.I)


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for mod in ("mlpilot", "sklearn", "polars", "numpy", "pandas"):
        try:
            if mod == "mlpilot":
                from importlib.metadata import version
                out[mod] = version("dataspoof-mlpilot")
            else:
                out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = "unknown"
    return out


def _fmt(v, digits=4):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{digits}f}" if abs(v) < 1e5 else f"{v:.4g}"
    return str(v)


def _pct(v):
    return "–" if v is None else f"{v:.1%}"


# ── building the card ───────────────────────────────────────────────────────

@dataclasses.dataclass
class ModelCard:
    """A model card: ``to_markdown()``, ``to_html()``, ``to_dict()`` / ``to_json()``, ``save(path)``."""

    data: dict

    @property
    def limitations(self) -> list:
        return self.data["limitations"]

    @property
    def ethical_considerations(self) -> list:
        return self.data["ethical_considerations"]

    def to_dict(self) -> dict:
        return jsonable(self.data)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        return _render_markdown(_blocks(self.data))

    def to_html(self) -> str:
        return _render_html(self.data["title"], _blocks(self.data))

    def save(self, path) -> Path:
        """Write to ``.md``, ``.html`` or ``.json`` (chosen by the file extension)."""
        path = Path(path)
        writers = {".md": self.to_markdown, ".markdown": self.to_markdown, ".html": self.to_html,
                   ".htm": self.to_html, ".json": self.to_json}
        if path.suffix.lower() not in writers:
            raise ValueError("Save a model card as .md, .html or .json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(writers[path.suffix.lower()](), encoding="utf-8")
        return path

    def _repr_html_(self):                      # rich display in notebooks
        return self.to_html()

    def __repr__(self) -> str:
        d = self.data["details"]
        return (f"ModelCard({self.data['title']!r}: {d['task']} '{d['target']}', "
                f"{len(self.limitations)} limitation(s), {len(self.ethical_considerations)} ethical note(s))")


def _automl_details(automl) -> dict:
    from ..automl.automl import _metric_label, _shown
    label = _metric_label(automl.metric_)
    best = automl.best_
    per_family: dict = {}
    for t in automl.trials_:
        if t["error"] is None and t["score"] > per_family.get(t["model"], {"score": -np.inf})["score"]:
            per_family[t["model"]] = t
    board = [{"model": t["model"], label: round(_shown(automl.metric_, t["score"]), 4),
              "std": round(t["std"], 4)} for t in sorted(per_family.values(), key=lambda t: -t["score"])]
    baseline = per_family.get("baseline")
    members = None
    if best["model"] == "ensemble":
        members = [m["model"] for m in best["params"]["members"]]
    kinds = automl.feature_kinds_
    steps = [s.name for s in automl.preprocess_.steps] if automl.preprocess_ is not None else []
    return {
        "algorithm": best["model"], "ensemble_members": members, "hyperparameters": best["params"],
        "metric": label, "metric_id": automl.metric_, "cv": automl.cv_description_,
        "cv_score": round(_shown(automl.metric_, best["score"]), 6), "cv_std": round(best["std"], 6),
        "baseline_score": round(_shown(automl.metric_, baseline["score"]), 6) if baseline else None,
        "n_trials": len(automl.trials_), "fit_seconds": automl.fit_seconds_,
        "leaderboard": board, "time_series": automl.task_.time_series, "time_col": automl.task_.time_col,
        "task_reason": automl.task_.reason,
        "training_data": {"rows": automl.n_rows_, "features": automl.n_features_,
                          "numeric_features": sum(k == "num" for k in kinds.values()),
                          "categorical_features": sum(k == "cat" for k in kinds.values()),
                          "feature_names": list(kinds), "dropped_leaky": list(automl.dropped_leaky_),
                          "leakage_findings": (automl.leakage_.to_dict()["findings"]
                                               if getattr(automl, "leakage_", None) is not None else None),
                          "preprocessing": steps},
    }


def model_card(model, test_df=None, *, target: str | None = None, sensitive=None, name: str | None = None,
               version: str = "1.0", description: str | None = None, intended_use: str | None = None,
               out_of_scope: str | None = None, owners=None, limitations=None, ethical_considerations=None,
               pos_label=None, explain_rows: int = 100, quality: bool = True, seed: int = 42) -> ModelCard:
    """Generate a model card for a fitted ``AutoML`` model (or scikit-learn estimator + ``target``).

    Model details, training data, cross-validation leaderboard and settings come from the
    model itself. With labelled ``test_df`` the card adds held-out metrics with confidence
    intervals, calibration, SHAP feature importance, error analysis, a data-quality score,
    and — given ``sensitive`` columns — a fairness review. Limitations and ethical
    considerations are generated from what was found; add your own with ``limitations`` /
    ``ethical_considerations`` and describe the purpose with ``intended_use``.
    """
    from .errors import error_analysis
    from .explain import _explain_X, _sample, explain
    from .fairness import fairness_report
    from .metrics import evaluate_model

    ad = ModelAdapter.of(model, target)
    sensitive = [sensitive] if isinstance(sensitive, str) else list(sensitive or [])
    automl = ad.automl
    info = _automl_details(automl) if automl is not None else {}
    title = name or f"{ad.target} {ad.task} model"
    data: dict[str, Any] = {
        "title": title,
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "details": {"name": title, "version": version, "description": description,
                    "owners": [owners] if isinstance(owners, str) else list(owners or []),
                    "task": ad.task, "target": ad.target, "classes": ad.classes, "model": ad.name,
                    **{k: info.get(k) for k in ("algorithm", "ensemble_members", "hyperparameters", "metric",
                                                "cv", "cv_score", "cv_std", "baseline_score", "n_trials",
                                                "fit_seconds", "time_series", "time_col", "task_reason")}},
        "intended_use": {"primary": intended_use, "out_of_scope": out_of_scope},
        "training_data": info.get("training_data"),
        "leaderboard": info.get("leaderboard"),
        "evaluation": None, "explainability": None, "error_analysis": None, "fairness": None,
        "data_quality": None, "versions": _versions(),
    }
    limits = list(limitations or [])
    ethics = list(ethical_considerations or [])
    evaluation = errors = fairness = None

    if test_df is not None:
        test = ad.labelled(as_polars(test_df))
        evaluation = evaluate_model(ad, test, pos_label=pos_label, seed=seed)
        data["evaluation"] = evaluation.to_dict(curves=False)
        errors = error_analysis(ad, test)
        data["error_analysis"] = {
            "metric": errors.metric, "overall": errors.overall, "min_size": errors.min_size,
            "problem_slices": errors.problem_slices.head(10).to_dicts(),
            "weakest_slices": errors.slices.head(5).to_dicts(),
            "confusions": errors.confusions.head(5).to_dicts() if errors.confusions is not None else None}
        exp = explain(ad, test, rows=explain_rows, pos_label=pos_label, seed=seed)
        if quality:
            from ..quality import check_quality
            q = check_quality(test)
            data["data_quality"] = {"score": q.score, "grade": q.grade, "dimensions": q.dimensions,
                                    "issues": len(q.issues),
                                    "top_issues": [i.title for i in q.issues[:5]]}
        if sensitive:
            fairness = fairness_report(ad, test, sensitive, pos_label=pos_label)
            data["fairness"] = fairness.to_dict()
    elif automl is not None and getattr(automl, "_importance_sample", None) is not None:
        X = automl._importance_sample[0]
        exp = _explain_X(ad, _sample(X, explain_rows, seed), _sample(X, 20, seed + 1), pos_label=pos_label,
                         seed=seed)
    else:
        exp = None
    if exp is not None:
        imp = exp.global_importance().head(15)
        data["explainability"] = {"method": exp.method, "output": exp.output, "rows": len(exp.values),
                                  "base_value": exp.base_value, "top_features": imp.to_dicts()}

    # ── generated limitations / ethical considerations ──────────────────────
    d = data["details"]
    if evaluation is None:
        limits.append("Not evaluated on held-out data: the scores above are cross-validation estimates. "
                      "Generate the card with test_df=... for held-out metrics.")
    else:
        m = evaluation.metrics
        if evaluation.n_rows < 200:
            limits.append(f"The evaluation set has only {evaluation.n_rows} labelled rows, so the metrics are "
                          "uncertain — see the 95% confidence intervals.")
        if d.get("metric") and d.get("cv_score") is not None and m.get(d["metric"]) is not None:
            cv, held = d["cv_score"], m[d["metric"]]
            lower_better = info.get("metric_id", "").startswith("neg_")
            worse = (held > cv * 1.15) if lower_better else (cv - held > 0.05)
            if worse:
                limits.append(f"Held-out {d['metric']} ({held:.4f}) is clearly worse than in cross-validation "
                              f"({cv:.4f}): possible overfitting, or the evaluation data differs from the "
                              "training data.")
        ece = m.get("ece")
        if ece is not None and ece > 0.1:
            limits.append(f"Predicted probabilities are poorly calibrated (expected calibration error {ece:.2f}); "
                          "don't read them as exact likelihoods without recalibrating.")
        if evaluation.per_class is not None:
            support = evaluation.per_class["support"].to_numpy()
            if support.sum() and support.min() / support.sum() < 0.1:
                rare = evaluation.per_class["class"][int(np.argmin(support))]
                limits.append(f"Class '{rare}' is rare in the evaluation data ({support.min()} rows); its "
                              "per-class metrics are unreliable.")
        if errors is not None and errors.problem_slices.height:
            worst = errors.problem_slices.head(3)
            fmt = _pct if errors.metric == "error_rate" else _fmt
            parts = "; ".join(f"{r['slice']} ({fmt(r['error'])} vs {fmt(errors.overall)} overall)"
                              for r in worst.iter_rows(named=True))
            limits.append(f"Performs significantly worse on some segments: {parts}.")
        dq = data["data_quality"]
        if dq and dq["grade"] in ("C", "D", "F"):
            limits.append(f"The evaluation data has quality grade {dq['grade']} ({dq['score']:.0f}/100); "
                          "metrics on cleaner data may differ.")
    if info:
        if d["baseline_score"] is not None and d["cv_score"] is not None and d["algorithm"] != "baseline":
            lower_better = info["metric_id"].startswith("neg_")
            gain = (d["baseline_score"] - d["cv_score"]) if lower_better else (d["cv_score"] - d["baseline_score"])
            scale = abs(d["baseline_score"]) if lower_better and d["baseline_score"] else 1.0
            if gain / scale < 0.02:
                limits.append("The model barely beats a trivial baseline; the features may carry little signal.")
        if info["training_data"]["dropped_leaky"]:
            limits.append(f"Columns dropped as target leakage: {', '.join(info['training_data']['dropped_leaky'])}. "
                          "They were not used by the model.")
        if d.get("time_series"):
            limits.append(f"Trained on time-ordered data (by '{d['time_col']}'); expect performance to change as "
                          "conditions drift — monitor it and retrain regularly.")
        limits.append(f"Trained on {info['training_data']['rows']:,} rows; predictions are only reliable for data "
                      "resembling that training data. Monitor inputs for drift in production.")

    if fairness is not None:
        names = {"disparate_impact_ratio": "disparate impact", "demographic_parity_difference": "demographic parity gap",
                 "equal_opportunity_difference": "TPR gap", "equalized_odds_difference": "equalized-odds gap",
                 "predictive_parity_difference": "precision gap", "accuracy_difference": "accuracy gap",
                 "error_ratio": "error ratio", "bias_difference": "bias gap"}
        for r in fairness.results:
            if r.flags:
                flagged = [f"{names.get(f.split()[0], f.split()[0])} {f.split()[1]}" for f in r.flags]
                ethics.append(f"Fairness ({r.attribute}): {len(r.flags)} disparity flag(s) — {', '.join(flagged)}. "
                              "See the Fairness section; decide whether these gaps are acceptable for the "
                              "intended use.")
            if r.used_by_model and " & " not in r.attribute:
                ethics.append(f"The sensitive attribute '{r.attribute}' is an input to the model; check this is "
                              "lawful and appropriate for the intended use.")
        proxies = fairness.proxies.filter(fairness.proxies["flag"])
        for row in proxies.head(5).iter_rows(named=True):
            ethics.append(f"'{row['feature']}' is strongly associated with '{row['attribute']}' "
                          f"(Cramér's V {row['cramers_v']:.2f}) and may act as a proxy for it.")
        if fairness.passed:
            ethics.append(f"No disparities beyond the configured limits across: {', '.join(sensitive)}.")
    else:
        cols = list(info["training_data"]["feature_names"]) if info else []
        if test_df is not None:
            cols = [c for c in as_polars(test_df).columns if c != ad.target]
        hints = sorted({c for c in cols if SENSITIVE_HINTS.search(c)})
        msg = "No fairness review was run: pass sensitive=[...] to check for bias across groups."
        if hints:
            msg += f" Columns that may be sensitive: {', '.join(hints)}."
        ethics.append(msg)
    data["limitations"], data["ethical_considerations"] = limits, ethics
    return ModelCard(data)


# ── rendering ───────────────────────────────────────────────────────────────
# A card is a list of blocks rendered to Markdown or HTML, so both always say the same thing:
#   ("h2", text) ("h3", text) ("p", text) ("kv", [(k, v)]) ("table", headers, rows)
#   ("bars", [(label, value_text, fraction, note)]) ("list", [text]) ("matrix", headers, rows)

def _blocks(data: dict) -> list:
    b: list[tuple] = []
    d = data["details"]
    task = {"binary": "Binary classification", "multiclass": "Multiclass classification",
            "regression": "Regression"}.get(d["task"], d["task"])
    b.append(("p", f"Generated {data['generated']} by MLPilot "
                   f"{data['versions'].get('mlpilot', '')}"))
    if d.get("description"):
        b.append(("p", d["description"]))

    b.append(("h2", "Model details"))
    kv = [("Task", f"{task} of `{d['target']}`"), ("Version", d["version"])]
    if d.get("classes"):
        kv.append(("Classes", ", ".join(map(str, d["classes"]))))
    algo = d.get("algorithm") or d["model"]
    if d.get("ensemble_members"):
        algo += f" (soft-voting: {', '.join(d['ensemble_members'])})"
    kv.append(("Algorithm", algo))
    if d.get("hyperparameters") and not d.get("ensemble_members"):
        kv.append(("Hyperparameters", json.dumps(d["hyperparameters"], default=str) if d["hyperparameters"]
                   else "library defaults"))
    if d.get("metric"):
        kv.append(("Selection metric", f"{d['metric']} ({d['cv']})"))
        kv.append(("Cross-validated score", f"{_fmt(d['cv_score'])} ± {_fmt(d['cv_std'])}"))
        if d.get("baseline_score") is not None:
            kv.append(("Baseline score", _fmt(d["baseline_score"])))
        kv.append(("Search", f"{d['n_trials']} trials in {d['fit_seconds']}s"))
    if d.get("task_reason"):
        kv.append(("Task detection", d["task_reason"]))
    if d.get("owners"):
        kv.append(("Owners", ", ".join(d["owners"])))
    b.append(("kv", kv))

    b.append(("h2", "Intended use"))
    iu = data["intended_use"]
    b.append(("p", iu["primary"] or "_Not specified — describe the intended users and decisions this model "
                                     "supports (model_card(..., intended_use=...))._"))
    if iu.get("out_of_scope"):
        b.append(("p", f"**Out of scope:** {iu['out_of_scope']}"))

    td = data.get("training_data")
    if td:
        b.append(("h2", "Training data"))
        kv = [("Rows", f"{td['rows']:,}"),
              ("Features", f"{td['features']} ({td['numeric_features']} numeric, "
                           f"{td['categorical_features']} categorical)")]
        if td["dropped_leaky"]:
            kv.append(("Dropped (leakage)", ", ".join(td["dropped_leaky"])))
        if td["preprocessing"]:
            kv.append(("Preprocessing", " → ".join(td["preprocessing"])))
        names = td["feature_names"]
        kv.append(("Feature names", ", ".join(names[:40]) + (f" … (+{len(names) - 40})" if len(names) > 40 else "")))
        b.append(("kv", kv))

    if data.get("leaderboard"):
        b.append(("h2", "Model selection"))
        metric = d["metric"]
        rows = [[str(i), r["model"], _fmt(r[metric]), _fmt(r["std"])] for i, r in enumerate(data["leaderboard"], 1)]
        b.append(("table", ["#", "Model", f"CV {metric}", "± std"], rows))

    ev = data.get("evaluation")
    if ev:
        b.append(("h2", "Held-out evaluation"))
        b.append(("p", f"{ev['n_rows']:,} labelled rows" + (f"; positive class `{ev['positive_class']}`"
                                                             if ev.get("positive_class") is not None else "")))
        ci = ev.get("confidence_intervals") or {}
        rows = [[k, _fmt(v), (f"{_fmt(ci[k][0])} – {_fmt(ci[k][1])}" if k in ci else "")]
                for k, v in ev["metrics"].items() if v is not None]
        b.append(("table", ["Metric", "Value", "95% CI"], rows))
        if ev.get("confusion_matrix"):
            cm = ev["confusion_matrix"]
            heads = ["actual \\ predicted"] + [k[5:] for k in cm[0] if k != "actual"]
            b.append(("h3", "Confusion matrix"))
            b.append(("matrix", heads, [[r["actual"]] + [r[k] for k in r if k != "actual"] for r in cm]))
        if ev.get("per_class") and len(ev["per_class"]) <= 20:
            b.append(("h3", "Per class"))
            b.append(("table", ["Class", "Precision", "Recall", "F1", "Support"],
                      [[r["class"], _fmt(r["precision"], 3), _fmt(r["recall"], 3), _fmt(r["f1"], 3),
                        str(r["support"])] for r in ev["per_class"]]))
        if ev.get("calibration"):
            b.append(("h3", "Calibration"))
            b.append(("p", f"Expected calibration error: {_fmt(ev['metrics'].get('ece'), 3)} "
                           "(0 = predicted probabilities match observed frequencies)."))
            b.append(("table", ["Predicted", "Rows", "Mean predicted", "Observed"],
                      [[r["bin"], str(r["count"]), _fmt(r["mean_predicted"], 3), _fmt(r["observed"], 3)]
                       for r in ev["calibration"]]))
        if ev.get("thresholds"):
            t = ev["thresholds"]
            best = max((r for r in t if r["f1"] is not None), key=lambda r: r["f1"], default=None)
            if best:
                b.append(("p", f"Best F1 decision threshold: **{best['threshold']:.2f}** (F1 {best['f1']:.3f}, "
                               f"precision {_fmt(best['precision'], 3)}, recall {_fmt(best['recall'], 3)})."))
        if ev.get("residuals"):
            r = ev["residuals"]
            b.append(("p", f"Residuals (predicted − actual): mean {_fmt(r['mean'])}, std {_fmt(r['std'])}; "
                           f"90% within [{_fmt(r['p05'])}, {_fmt(r['p95'])}]."))
        if ev.get("error_by_range"):
            b.append(("h3", "Error by range of the actual value"))
            b.append(("table", ["Range", "Rows", "Mean actual", "Mean predicted", "MAE", "Bias"],
                      [[r["range"], str(r["count"]), _fmt(r["mean_actual"]), _fmt(r["mean_predicted"]),
                        _fmt(r["mae"]), _fmt(r["bias"])] for r in ev["error_by_range"]]))

    ex = data.get("explainability")
    if ex:
        b.append(("h2", "Explainability"))
        b.append(("p", f"Mean absolute SHAP value per feature ({ex['method']}, {ex['rows']} rows): the average "
                       f"contribution to {ex['output']}. Direction: how higher values of a numeric feature move it."))
        top = ex["top_features"]
        peak = max((r["mean_abs_shap"] for r in top), default=1) or 1
        bars = []
        for r in top:
            dct = r.get("direction")
            note = "" if dct is None or abs(dct) < 0.2 else ("higher → increases" if dct > 0 else "higher → decreases")
            bars.append((r["feature"], f"{r['mean_abs_shap']:.4f} ({r['share']:.0%})", r["mean_abs_shap"] / peak, note))
        b.append(("bars", bars))

    ea = data.get("error_analysis")
    if ea:
        b.append(("h2", "Error analysis"))
        fmt = _pct if ea["metric"] == "error_rate" else _fmt
        label = "Error rate" if ea["metric"] == "error_rate" else "MAE"
        b.append(("p", f"Overall {label.lower()}: {fmt(ea['overall'])}. Segments of at least {ea['min_size']} rows "
                       "tested against the rest (Benjamini–Hochberg corrected)."))
        shown = ea["problem_slices"] or ea["weakest_slices"]
        if not ea["problem_slices"]:
            b.append(("p", "No segment is significantly worse than average. Weakest segments:"))
        b.append(("table", ["Segment", "Rows", label, "× overall", "q-value"],
                  [[r["slice"], str(r["size"]), fmt(r["error"]), f"{r['lift']:.2f}", f"{r['q_value']:.3g}"]
                   for r in shown]))
        if ea.get("confusions"):
            b.append(("p", "Most common mistakes: " + ", ".join(
                f"{r['actual']} → {r['predicted']} ({r['count']})" for r in ea["confusions"])))

    fr = data.get("fairness")
    if fr:
        b.append(("h2", "Fairness"))
        b.append(("p", ("No disparities beyond the configured limits." if fr["passed"] else
                        f"{sum(len(r['flags']) for r in fr['results'])} disparity flag(s).")
                  + (f" Positive class: `{fr['positive_class']}`." if fr.get("positive_class") is not None else "")))
        for r in fr["results"]:
            b.append(("h3", r["attribute"]))
            groups = r["groups"]
            cols = [c for c in ("size", "base_rate", "selection_rate", "tpr", "fpr", "precision", "accuracy",
                                "mean_actual", "mean_prediction", "mae", "bias") if groups and c in groups[0]]
            pretty = {"size": "Rows", "base_rate": "Actual +", "selection_rate": "Predicted +", "tpr": "TPR",
                      "fpr": "FPR", "precision": "Precision", "accuracy": "Accuracy", "mean_actual": "Mean actual",
                      "mean_prediction": "Mean predicted", "mae": "MAE", "bias": "Bias"}
            b.append(("table", ["Group"] + [pretty[c] for c in cols],
                      [[str(g["group"])] + [(_fmt(g[c], 3) if c != "size" else str(g[c])) for c in cols]
                       for g in groups]))
            disp = [(k.replace("_", " "), _fmt(v, 3)) for k, v in r["disparities"].items() if v is not None]
            if disp:
                b.append(("kv", disp))
            if r["flags"] or r["notes"]:
                b.append(("list", [f"⚠ {f}" for f in r["flags"]] + r["notes"]))
        flagged = [p for p in fr["proxies"] if p["flag"]]
        if flagged:
            b.append(("h3", "Possible proxies"))
            b.append(("table", ["Feature", "Sensitive attribute", "Cramér's V"],
                      [[p["feature"], p["attribute"], _fmt(p["cramers_v"], 2)] for p in flagged[:10]]))

    dq = data.get("data_quality")
    if dq:
        b.append(("h2", "Evaluation data quality"))
        b.append(("p", f"Score **{dq['score']:.0f}/100 (grade {dq['grade']})**, {dq['issues']} issue(s)."
                  + (" Top: " + "; ".join(dq["top_issues"]) + "." if dq["top_issues"] else "")))

    b.append(("h2", "Limitations"))
    b.append(("list", data["limitations"] or ["None recorded."]))
    b.append(("h2", "Ethical considerations"))
    b.append(("list", data["ethical_considerations"] or ["None recorded."]))
    b.append(("h2", "Versions"))
    b.append(("kv", list(data["versions"].items())))
    return b


def _md_cell(v) -> str:
    return str(v).replace("|", "\\|").replace("\n", " ")


def _render_markdown(blocks) -> str:
    out = []
    title = None
    for blk in blocks:
        kind = blk[0]
        if kind == "h2":
            out.append(f"\n## {blk[1]}\n")
        elif kind == "h3":
            out.append(f"\n### {blk[1]}\n")
        elif kind == "p":
            out.append(blk[1] + "\n")
        elif kind == "kv":
            out.append("| | |\n|---|---|")
            out.extend(f"| **{_md_cell(k)}** | {_md_cell(v)} |" for k, v in blk[1])
            out.append("")
        elif kind in ("table", "matrix"):
            heads, rows = blk[1], blk[2]
            out.append("| " + " | ".join(map(_md_cell, heads)) + " |")
            out.append("|" + "---|" * len(heads))
            out.extend("| " + " | ".join(map(_md_cell, r)) + " |" for r in rows)
            out.append("")
        elif kind == "bars":
            out.append("| Feature | Importance | | Effect |\n|---|---|---|---|")
            for label, value, frac, note in blk[1]:
                bar = "█" * max(1, round(frac * 20))
                out.append(f"| {_md_cell(label)} | {value} | `{bar}` | {note} |")
            out.append("")
        elif kind == "list":
            out.extend(f"- {item}" for item in blk[1])
            out.append("")
    return "\n".join(out).strip() + "\n"


def _inline(text: str) -> str:
    """Escape, then apply the tiny bit of Markdown the card uses (**bold**, `code`, _italic_)."""
    t = html.escape(str(text))
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
    t = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<em>\1</em>", t)
    return t


_CSS = """
:root{--bg:#f7f8fa;--card:#fff;--text:#1d2330;--muted:#5d6676;--line:#e3e6eb;--accent:#2f6fde;
--accent-soft:rgba(47,111,222,.14);--warn:#b45309;--warn-soft:rgba(217,119,6,.12)}
@media (prefers-color-scheme:dark){:root{--bg:#0f1218;--card:#171b23;--text:#e6e9ef;--muted:#9aa3b2;
--line:#2a303b;--accent:#6b9cff;--accent-soft:rgba(107,156,255,.18);--warn:#f5a524;--warn-soft:rgba(245,165,36,.14)}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);
font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:980px;margin:0 auto;padding:32px 16px 64px}
header{margin-bottom:8px}h1{font-size:28px;margin:0 0 4px;letter-spacing:-.01em}
.sub{color:var(--muted);margin:0 0 20px}
section{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px 22px;margin:16px 0}
h2{font-size:18px;margin:0 0 12px}h3{font-size:15px;margin:18px 0 8px;color:var(--muted)}
p{margin:8px 0}code{background:var(--accent-soft);padding:1px 5px;border-radius:4px;font-size:13px}
.tbl{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600}table.kv th{width:32%;color:var(--text)}
.matrix td{text-align:center}.matrix td:first-child{text-align:left;font-weight:600}
.bars{display:grid;grid-template-columns:minmax(120px,240px) 1fr auto;gap:6px 12px;align-items:center;font-size:14px}
.bar{height:10px;border-radius:5px;background:var(--accent)}.track{background:var(--accent-soft);border-radius:5px}
.val{color:var(--muted);white-space:nowrap}.note{grid-column:2/4;color:var(--muted);font-size:12px;margin-top:-4px}
ul{margin:8px 0;padding-left:20px}li{margin:4px 0}li.warn{color:var(--warn)}
@media (max-width:600px){.bars{grid-template-columns:1fr auto}.bars .track{grid-column:1/3}.note{grid-column:1/3}}
"""


def _render_html(title: str, blocks, doc_kind: str = "Model card") -> str:
    parts: list[str] = []
    open_section = False

    def close():
        nonlocal open_section
        if open_section:
            parts.append("</section>")
            open_section = False

    intro: list[str] = []
    for blk in blocks:
        kind = blk[0]
        if kind == "h2":
            close()
            parts.append(f"<section><h2>{_inline(blk[1])}</h2>")
            open_section = True
            continue
        target = parts if open_section else intro
        if kind == "h3":
            target.append(f"<h3>{_inline(blk[1])}</h3>")
        elif kind == "p":
            target.append(f"<p{' class=sub' if not open_section else ''}>{_inline(blk[1])}</p>")
        elif kind == "kv":
            rows = "".join(f"<tr><th>{_inline(k)}</th><td>{_inline(v)}</td></tr>" for k, v in blk[1])
            target.append(f"<div class=tbl><table class=kv>{rows}</table></div>")
        elif kind == "table":
            head = "".join(f"<th>{_inline(h)}</th>" for h in blk[1])
            body = "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in blk[2])
            target.append(f"<div class=tbl><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>")
        elif kind == "matrix":
            head = "".join(f"<th>{_inline(h)}</th>" for h in blk[1])
            values = [v for r in blk[2] for v in r[1:]]
            peak = max(values) if values else 1
            body = ""
            for i, r in enumerate(blk[2]):
                cells = "".join(
                    f"<td style=\"background:rgba(47,111,222,{0.08 + 0.6 * v / peak if peak else 0:.2f})"
                    f"{';font-weight:700' if j == i else ''}\">{v}</td>"
                    for j, v in enumerate(r[1:]))
                body += f"<tr><td>{_inline(r[0])}</td>{cells}</tr>"
            target.append(f"<div class=tbl><table class=matrix><thead><tr>{head}</tr></thead>"
                          f"<tbody>{body}</tbody></table></div>")
        elif kind == "bars":
            bar_rows: list[str] = []
            for label, value, frac, note in blk[1]:
                bar_rows.append(f"<div>{_inline(label)}</div><div class=track><div class=bar "
                            f"style=\"width:{max(2, frac * 100):.1f}%\"></div></div><div class=val>{_inline(value)}</div>")
                if note:
                    bar_rows.append(f"<div class=note>{_inline(note)}</div>")
            target.append(f"<div class=bars>{''.join(bar_rows)}</div>")
        elif kind == "list":
            items = "".join(f"<li{' class=warn' if str(i).startswith('⚠') else ''}>{_inline(i)}</li>"
                            for i in blk[1])
            target.append(f"<ul>{items}</ul>")
    close()
    return ("<!doctype html><html lang=en><head><meta charset=utf-8>"
            "<meta name=viewport content=\"width=device-width,initial-scale=1\">"
            f"<title>{html.escape(title)} · {html.escape(doc_kind)}</title><style>{_CSS}</style></head><body><main>"
            f"<header><h1>{html.escape(title)}</h1>{''.join(intro)}</header>{''.join(parts)}</main></body></html>")
