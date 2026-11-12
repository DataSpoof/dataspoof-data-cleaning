# Statistical testing: API reference

Generated from the source docstrings. For a task-oriented introduction, read the [Statistical testing guide](../guide/stats.md).

**On this page:**

- [`mlpilot.stats`](#mlpilotstats): `TestResult`, `effect_label`, `adjust_pvalues`, `normality_test`, `ttest`, `mann_whitney`, `paired_test`, `anova`, `kruskal`, `compare_groups`, `chi_square`, `correlation_test`, `proportion_test`, `ab_test` …

---

## `mlpilot.stats`

<sub>Source: [`mlpilot/stats.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/stats.py)</sub>

Statistical testing toolkit: the right test for the data, with effect sizes, confidence
intervals, assumption checks and a plain-English reading — not just a p-value.

* `compare_groups(df, value="revenue", group="region")`  picks Welch's t-test / Mann-Whitney
  (2 groups) or ANOVA / Welch-type ANOVA / Kruskal-Wallis (3+), checks normality and equal
  variances first, adds effect sizes and post-hoc pairwise comparisons
* `ttest`, `mann_whitney`, `anova`, `kruskal`, `paired_test`
* `chi_square(df, "plan", "churned")`   contingency test with Cramér's V; Fisher's exact for small 2x2
* `correlation_test(df, "x", "y")`      Pearson / Spearman / Kendall with a confidence interval
* `normality_test`, `proportion_test` (two-proportion z-test), `ab_test` (conversion or
  continuous metric: lift, CI, p-value), `required_sample_size` (power analysis)
* `adjust_pvalues` (Holm / Benjamini-Hochberg / Bonferroni) and `test_all(df, target)` — every
  column's association with a target, corrected for multiple testing

### class `TestResult`

```python
class TestResult:
```

| Field | Type | Default |
|---|---|---|
| `test` | `str` | required |
| `statistic` | `float` | required |
| `p_value` | `float` | required |
| `effect_size` | `float \| None` | `None` |
| `effect_name` | `str \| None` | `None` |
| `ci` | `tuple \| None` | `None` |
| `n` | `int \| None` | `None` |
| `alpha` | `float` | `0.05` |
| `assumptions` | `dict` | `dict()` |
| `details` | `dict` | `dict()` |
| `interpretation` | `str` | `''` |

#### `TestResult.significant` *(property)*

```python
significant: bool
```

#### `TestResult.summary()`

```python
summary() -> str
```

#### `TestResult.to_dict()`

```python
to_dict() -> dict
```

### `effect_label()`

```python
def effect_label(name: str | None, value: float | None) -> str
```

Conventional size labels (Cohen 1988 and common practice).

### `adjust_pvalues()`

```python
def adjust_pvalues(pvalues, method: str='holm') -> list
```

Multiple-testing correction: `holm` (default, controls family-wise error), `bh`
(Benjamini-Hochberg false-discovery rate) or `bonferroni`.

### `normality_test()`

```python
def normality_test(x, alpha: float=0.05) -> TestResult
```

Shapiro-Wilk (n <= 5000) or D'Agostino-Pearson (larger samples).

### `ttest()`

```python
def ttest(a, b, *, equal_var: bool=False, alpha: float=0.05, names=('A', 'B')) -> TestResult
```

Two-sample t-test (Welch's by default — safe when variances differ) with Hedges' g and a CI
for the difference in means.

### `mann_whitney()`

```python
def mann_whitney(a, b, *, alpha: float=0.05, names=('A', 'B')) -> TestResult
```

Mann-Whitney U (no normality assumption) with the rank-biserial correlation.

### `paired_test()`

```python
def paired_test(before, after, *, alpha: float=0.05) -> TestResult
```

Paired t-test if the differences look normal, otherwise the Wilcoxon signed-rank test.

### `anova()`

```python
def anova(*groups, names=None, alpha: float=0.05, equal_var: bool | None=None) -> TestResult
```

One-way ANOVA (equal variances) or Alexander-Govern (unequal variances — the robust choice),
with eta-squared / omega-squared and Tukey HSD pairwise comparisons.

### `kruskal()`

```python
def kruskal(*groups, names=None, alpha: float=0.05) -> TestResult
```

Kruskal-Wallis (no normality assumption) with epsilon-squared and Holm-corrected pairwise
Mann-Whitney tests.

### `compare_groups()`

```python
def compare_groups(df: pl.DataFrame, value: str, group: str, *, alpha: float=0.05, parametric: bool | None=None, min_group: int=3) -> TestResult
```

Compare a numeric `value` across the levels of `group`, choosing the test for you:
normality per group decides parametric vs rank-based, Levene decides equal variances.

### `chi_square()`

```python
def chi_square(df: pl.DataFrame, a: str, b: str, *, alpha: float=0.05) -> TestResult
```

Independence of two categorical columns; Fisher's exact test for 2x2 tables with small
expected counts. Effect size: Cramér's V (bias-corrected).

### `correlation_test()`

```python
def correlation_test(df: pl.DataFrame, x: str, y: str, *, method: str='pearson', alpha: float=0.05) -> TestResult
```

Correlation with a p-value and (Pearson/Spearman) a Fisher-z confidence interval.

### `proportion_test()`

```python
def proportion_test(successes_a: int, total_a: int, successes_b: int, total_b: int, *, alpha: float=0.05) -> TestResult
```

Two-proportion z-test with a Newcombe-style CI for the difference.

### `ab_test()`

```python
def ab_test(df: pl.DataFrame, variant: str, metric: str, *, control=None, alpha: float=0.05) -> TestResult
```

A/B test of a binary (conversion) or numeric metric between two variants; reports the lift
of the treatment over `control` (default: the first variant alphabetically).

### `required_sample_size()`

```python
def required_sample_size(*, baseline_rate: float | None=None, mde: float, std: float | None=None, alpha: float=0.05, power: float=0.8) -> int
```

Rows needed **per variant** to detect an absolute effect `mde`: on a conversion rate
(give `baseline_rate`) or on a mean (give the metric's `std`).

### `test_all()`

```python
def test_all(df: pl.DataFrame, target: str, *, alpha: float=0.05, correction: str='bh', max_categories: int=30) -> pl.DataFrame
```

Every column's association with `target` using the appropriate test (numeric target:
correlation / group comparison; categorical target: group comparison / chi-square), with
p-values corrected for multiple testing. Sorted by effect size.
