# Causal inference & uplift: API reference

Generated from the source docstrings. For a task-oriented introduction, read the [Causal inference & uplift guide](../guide/causal.md).

**On this page:**

- [`mlpilot.causal`](#mlpilotcausal): `EffectEstimate`, `estimate_effect`, `UpliftModel`, `qini`

---

## `mlpilot.causal`

<sub>Source: [`mlpilot/causal.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/causal.py)</sub>

Causal effects and uplift: what a treatment (discount, campaign, feature) actually *causes*.

`estimate_effect(df, treatment="got_discount", outcome="revenue")` estimates the average
treatment effect with **AIPW** (augmented inverse-propensity weighting, "doubly robust"): models
of the outcome and of who got treated are fitted with cross-fitting, so the estimate stays
consistent if either model is right. It reports a 95% confidence interval, the naive
(confounded) difference for comparison, and diagnostics that tell you when not to trust it:
**overlap** (treated and untreated must be comparable) and **covariate balance** after weighting.

`UpliftModel` estimates the effect *per customer* (T-learner) so you can target those who
respond; `qini` scores how well a ranking finds them.

Causal estimates rest on an assumption no data can prove: every common cause of treatment and
outcome is among the `confounders`. The report says so.

### class `EffectEstimate`

```python
class EffectEstimate:
```

| Field | Type | Default |
|---|---|---|
| `treatment` | `str` | required |
| `outcome` | `str` | required |
| `ate` | `float` | required |
| `ci` | `tuple` | required |
| `p_value` | `float` | required |
| `naive_difference` | `float` | required |
| `n` | `int` | required |
| `n_treated` | `int` | required |
| `overlap` | `dict` | required |
| `balance` | `pl.DataFrame` | required |
| `confounders` | `list` | required |
| `method` | `str` | `'AIPW (doubly robust, cross-fitted)'` |

#### `EffectEstimate.significant` *(property)*

```python
significant: bool
```

#### `EffectEstimate.summary()`

```python
summary() -> str
```

### `estimate_effect()`

```python
def estimate_effect(df, treatment: str, outcome: str, *, confounders=None, folds: int=5, clip: float=0.01, seed: int=42) -> EffectEstimate
```

Average treatment effect of a binary `treatment` on `outcome` (numeric or binary).

`confounders` default to every other column. Propensities are clipped to
[`clip`, 1 - `clip`] to keep weights finite.

### class `UpliftModel`

```python
class UpliftModel:
```

Per-row treatment effect (T-learner: separate outcome models for treated and untreated).

```python
up = UpliftModel("got_offer", "converted").fit(campaign)
up.predict_uplift(customers)      # who gains most from the offer
```

**Constructor**

```python
UpliftModel(treatment: str, outcome: str, *, features=None, seed: int=42)
```

#### `UpliftModel.fit()`

```python
fit(df) -> 'UpliftModel'
```

#### `UpliftModel.predict_uplift()`

```python
predict_uplift(df) -> pl.Series
```

### `qini()`

```python
def qini(uplift, treatment, outcome, bins: int=10) -> dict
```

Qini curve and coefficient: how much more of the treatment effect is captured by targeting
in `uplift` order than at random (0 = no better than random; higher is better).
