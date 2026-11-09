"""Model security, part 1: how easily can a model's predictions be pushed around?

>>> rep = robustness_report(model, test_df)          # AutoML model, or a sklearn model with target=
>>> print(rep.summary())
>>> rep.features                                     # per-feature sensitivity table
>>> rep.passed(max_noise_flip=0.05, min_median_distance=0.25)

Black-box tests (only ``predict``/``predict_proba`` are used, so any model works, and the full
preprocessing is part of what is attacked):

* **Noise stability**: numeric inputs get Gaussian noise of 1%, 5% and 10% of each column's
  standard deviation; how often does the prediction change?
* **Minimal adversarial perturbation** (evasion attack): for each sampled row, a greedy
  coordinate search moves the most effective numeric feature step by step until the predicted
  class flips (classification) or the prediction moves by ``regression_shift`` standard
  deviations. The distance needed is reported in standard-deviation units (L-infinity); a model
  that flips at 0.05 sd can be fooled by changes nobody would notice.
* **Feature sensitivity**: flip rate when one feature is replaced by another observed value
  (categoricals), shifted by +-1 sd (numerics) or set to missing.
* **Extreme inputs**: values far outside the training range must not crash the model or produce
  NaN / infinite outputs.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import polars as pl

from .evaluation.adapter import ModelAdapter, as_polars


@dataclasses.dataclass
class RobustnessReport:
    task: str
    rows: int
    noise: dict                      # {noise level (sd): flip rate or relative change}
    attack: dict                     # minimal perturbation results
    features: pl.DataFrame           # per-feature sensitivity
    extreme: dict                    # {"crashed": [...], "non_finite": [...]}
    notes: list = dataclasses.field(default_factory=list)

    def passed(self, *, max_noise_flip: float = 0.05, min_median_distance: float = 0.25) -> bool:
        """True when 5%-noise flips at most ``max_noise_flip`` of predictions, the median
        adversarial distance is at least ``min_median_distance`` sd, and extreme inputs are handled."""
        noise_ok = self.noise.get(0.05, 0.0) <= max_noise_flip
        med = self.attack.get("median_distance")
        attack_ok = med is None or med >= min_median_distance
        return noise_ok and attack_ok and not self.extreme["crashed"] and not self.extreme["non_finite"]

    def summary(self) -> str:
        unit = "flip rate" if self.task != "regression" else "mean |change| / sd(prediction)"
        lines = [f"Robustness ({self.task}, {self.rows} rows)",
                 "  noise " + ", ".join(f"{int(k * 100)}% sd -> {v:.1%} {unit}" for k, v in self.noise.items())]
        a = self.attack
        if a.get("attacked"):
            lines.append(f"  adversarial: {a['success_rate']:.0%} of {a['attacked']} rows changed within "
                         f"{a['budget']} sd; median distance {a['median_distance']:.3g} sd"
                         + (f"; most used feature: {a['most_used_feature']}" if a.get("most_used_feature") else ""))
            lines.append("  rows fooled within: " + ", ".join(f"{k} sd -> {v:.0%}" for k, v in a["within"].items()))
        top = self.features.head(5)
        if top.height:
            lines.append("  most sensitive features: " + ", ".join(
                f"{r['feature']} ({r['flip_rate']:.0%})" for r in top.iter_rows(named=True)))
        if self.extreme["crashed"] or self.extreme["non_finite"]:
            lines.append(f"  extreme inputs: crashed on {self.extreme['crashed']}, "
                         f"non-finite output for {self.extreme['non_finite']}")
        else:
            lines.append("  extreme inputs: handled")
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


class _Probe:
    """Scores perturbed copies of rows and turns outputs into a comparable signal."""

    def __init__(self, adapter: ModelAdapter, regression_scale: float, schema: dict | None = None):
        self.a, self.scale, self.schema = adapter, regression_scale, schema or {}

    def _restore(self, df: pl.DataFrame) -> pl.DataFrame:
        """Perturbed columns back to the training dtypes (integers are rounded), so the model
        sees inputs of the same types it was trained on."""
        fix = []
        for c, t in self.schema.items():
            if c in df.columns and df.schema[c] != t:
                e = pl.col(c).round() if t.is_integer() else pl.col(c)
                fix.append(e.cast(t, strict=False).alias(c))
        return df.with_columns(fix) if fix else df

    def labels(self, df: pl.DataFrame) -> np.ndarray:
        return self.a.predict(self.a.X(self._restore(df)))

    def scores(self, df: pl.DataFrame, ref_labels: np.ndarray | None = None) -> np.ndarray:
        """Classification: P(reference class); regression: the prediction."""
        X = self.a.X(self._restore(df))
        if self.a.is_classification:
            p = self.a.proba(X)
            return p[np.arange(len(p)), ref_labels]
        return np.asarray(self.a.predict(X), dtype=float)

    def changed(self, base, new) -> np.ndarray:
        if self.a.is_classification:
            return np.asarray(base) != np.asarray(new)
        return np.abs(np.asarray(new, float) - np.asarray(base, float)) / self.scale


def _numeric(df: pl.DataFrame, exclude) -> list:
    return [c for c, t in df.schema.items() if c not in exclude and t.is_numeric() and df[c].drop_nulls().len()
            and (df[c].std() or 0) > 0]


def robustness_report(model, df, *, target: str | None = None, n: int = 200, noise=(0.01, 0.05, 0.1),
                      budget: float = 2.0, step: float = 0.05, regression_shift: float = 0.5, seed: int = 0,
                      features: list | None = None) -> RobustnessReport:
    """Run the black-box robustness tests on up to ``n`` rows of ``df`` (see the module docstring).
    ``budget`` caps the adversarial search (in sd units); ``step`` is its resolution."""
    a = ModelAdapter.of(model, target)
    df = as_polars(df)
    if a.target in df.columns:
        df = df.drop(a.target)
    rng = np.random.default_rng(seed)
    sample = df.sample(min(n, df.height), seed=seed) if df.height > n else df
    num = [c for c in (features or _numeric(df, [])) if c in _numeric(df, [])]
    cat = [c for c in (features or df.columns) if c in df.columns and c not in num and df[c].n_unique() > 1]
    sd = {c: float(df[c].std()) for c in num}
    schema = dict(sample.schema)
    base_labels = _Probe(a, 1.0, schema).labels(sample)
    if a.is_classification:
        probe = _Probe(a, 1.0, schema)
    else:
        probe = _Probe(a, float(np.std(base_labels)) or 1.0, schema)
    notes = []
    if not num:
        notes.append("no numeric features: noise and adversarial tests were skipped")

    # noise stability
    noise_out: dict = {}
    for level in noise:
        if not num:
            break
        noisy = sample.with_columns([(pl.col(c) + pl.Series(rng.normal(0, level * sd[c], sample.height)))
                                     .alias(c) for c in num])
        ch = probe.changed(base_labels, probe.labels(noisy))
        noise_out[level] = float(np.mean(ch))

    # minimal adversarial perturbation: greedy coordinate search, batched over rows
    attack: dict[str, Any] = (_attack(probe, a, sample, num, sd, base_labels, budget, step, regression_shift)
                              if num else {"attacked": 0})

    # per-feature sensitivity
    rows = []
    for c in num + cat:
        res = {"feature": c, "kind": "numeric" if c in num else "categorical"}
        if c in num:
            up = sample.with_columns((pl.col(c) + sd[c]).alias(c))
            down = sample.with_columns((pl.col(c) - sd[c]).alias(c))
            flips = np.maximum(probe.changed(base_labels, probe.labels(up)), probe.changed(base_labels, probe.labels(down)))
        else:
            values = df[c].drop_nulls().unique().to_list()
            cur = sample[c].to_list()
            swapped = [rng.choice([v for v in values if v != x]) if len(values) > 1 else x for x in cur]
            flips = probe.changed(base_labels, probe.labels(sample.with_columns(
                pl.Series(c, swapped, dtype=sample.schema[c]))))
        try:
            missing = probe.changed(base_labels, probe.labels(sample.with_columns(pl.lit(None, sample.schema[c]).alias(c))))
            res["missing_flip_rate"] = float(np.mean(missing > (0 if a.is_classification else 0.5)))
        except Exception:
            res["missing_flip_rate"] = None
            notes.append(f"the model fails when '{c}' is missing")
        res["flip_rate"] = float(np.mean(flips > (0 if a.is_classification else 0.5)))
        rows.append(res)
    feats = (pl.DataFrame(rows, schema={"feature": pl.String, "kind": pl.String, "flip_rate": pl.Float64,
                                        "missing_flip_rate": pl.Float64})
             .sort("flip_rate", descending=True) if rows else
             pl.DataFrame(schema={"feature": pl.String, "kind": pl.String, "flip_rate": pl.Float64,
                                  "missing_flip_rate": pl.Float64}))
    if attack.get("usage"):
        use = pl.DataFrame({"feature": list(attack["usage"]), "attack_uses": list(attack["usage"].values())})
        feats = feats.join(use, on="feature", how="left").with_columns(pl.col("attack_uses").fill_null(0))

    # extreme inputs
    crashed, non_finite = [], []
    probe_rows = sample.head(min(20, sample.height))
    for c in num:
        for mult in (-1e3, 1e3):
            try:
                X = a.X(probe._restore(probe_rows.with_columns(
                    (pl.lit(float(df[c].mean() or 0)) + mult * sd[c]).alias(c))))
                out = a.proba(X) if a.is_classification else np.asarray(a.predict(X), dtype=float)
                if not np.all(np.isfinite(out)):
                    non_finite.append(c)
                    break
            except Exception:
                crashed.append(c)
                break
    return RobustnessReport(task=a.task, rows=sample.height, noise=noise_out, attack=attack, features=feats,
                            extreme={"crashed": crashed, "non_finite": non_finite}, notes=notes)


def _attack(probe: _Probe, a: ModelAdapter, sample: pl.DataFrame, num: list, sd: dict, base_labels, budget: float,
            step: float, regression_shift: float) -> dict:
    """Greedy, decision-based L-inf attack in sd units. Each iteration tries +-step on every
    numeric feature for every unfinished row (one batched model call) and keeps the move that
    lowers the reference-class probability (or moves the regression output) the most."""
    n = sample.height
    cur = sample.with_columns([pl.col(c).cast(pl.Float64) for c in num])
    offset = np.zeros((n, len(num)))
    done = np.zeros(n, bool)
    dist = np.full(n, np.nan)
    usage = {c: 0 for c in num}
    ref: Any = base_labels if a.is_classification else None
    base_score = probe.scores(cur, ref)
    max_iter = int(np.ceil(budget / step))
    for _ in range(max_iter):
        todo = np.flatnonzero(~done)
        if not len(todo):
            break
        sub = cur[todo]
        cands = []
        for j, c in enumerate(num):
            for sign in (1.0, -1.0):
                cands.append((j, sign, sub.with_columns((pl.col(c) + sign * step * sd[c]).alias(c))))
        batch = pl.concat([x[2] for x in cands], how="vertical_relaxed")
        if a.is_classification:
            scores = probe.scores(batch, np.tile(ref[todo], len(cands))).reshape(len(cands), len(todo))
            labels = probe.labels(batch).reshape(len(cands), len(todo))
            # prefer a move that flips; otherwise the one that lowers P(reference class) most
            flipped = labels != ref[todo][None, :]
            objective = np.where(flipped, -np.inf, scores)
        else:
            vals = probe.scores(batch).reshape(len(cands), len(todo))
            moved = np.abs(vals - base_score[todo][None, :]) / probe.scale
            objective = -moved
            flipped = moved >= regression_shift
        best = np.argmin(objective, axis=0)
        new_rows = []
        for k, row in enumerate(todo):
            j, sign, frame = cands[best[k]]
            offset[row, j] += sign * step
            usage[num[j]] += 1
            new_rows.append(frame[k])
            if flipped[best[k], k]:
                done[row] = True
                dist[row] = float(np.max(np.abs(offset[row])))
        upd = pl.concat(new_rows, how="vertical_relaxed")
        cur = pl.concat([cur.with_row_index("_i").filter(~pl.col("_i").is_in(todo.tolist())),
                         upd.with_columns(pl.Series("_i", todo.astype(np.uint32)))], how="diagonal_relaxed") \
            .sort("_i").drop("_i").select(sample.columns)
    found = dist[~np.isnan(dist)]
    within = {f"{e:g}": (float(np.sum(~np.isnan(dist) & (dist <= e + 1e-9)) / n) if n else 0.0)
              for e in (0.1, 0.25, 0.5, 1.0)}
    return {"attacked": n, "budget": budget, "success_rate": float(len(found) / n) if n else 0.0,
            "median_distance": float(np.median(found)) if len(found) else None, "within": within,
            "usage": usage, "most_used_feature": max(usage, key=lambda k: usage[k]) if any(usage.values()) else None}
