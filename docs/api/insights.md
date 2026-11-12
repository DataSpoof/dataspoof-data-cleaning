# Insights & reporting: API reference

Generated from the source docstrings. For a task-oriented introduction, read the [Insights & reporting guide](../guide/insights.md).

**On this page:**

- [`mlpilot.insights`](#mlpilotinsights)
- [`mlpilot.insights.finder`](#mlpilotinsightsfinder): `Insight`, `find_insights`
- [`mlpilot.insights.narrate`](#mlpilotinsightsnarrate): `template_narrative`, `narrate`
- [`mlpilot.insights.report`](#mlpilotinsightsreport): `InsightsReport`, `insights_report`

---

## `mlpilot.insights`

<sub>Source: [`mlpilot/insights/__init__.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/insights/__init__.py)</sub>

Insights & reporting (module 10): plain-English findings and executive reports.

* `find_insights(df, target)`      ranked, significance-tested findings: drivers ("sales is 18%
  lower when stock < 20"), trends, correlations, concentration,
  outliers, data-quality notes, model drivers.
* `insights_report(df, target, llm=True)`   KPIs + findings + an executive narrative written
  by Claude (or a template, offline) -> .html / .md / .json.

---

## `mlpilot.insights.finder`

<sub>Source: [`mlpilot/insights/finder.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/insights/finder.py)</sub>

Automatic insights: statistically supported findings phrased in plain English.

Every insight carries the numbers behind it (segment size, values, p-value), so a report or
an LLM narrative can only restate what the data shows. Findings are associations, not causes.

### class `Insight`

```python
class Insight:
```

| Field | Type | Default |
|---|---|---|
| `kind` | `str` | required |
| `headline` | `str` | required |
| `strength` | `float` | required |
| `columns` | `list` | required |
| `evidence` | `dict` | required |

### `find_insights()`

```python
def find_insights(df, target: str | None=None, *, time_col='auto', model=None, max_insights: int=12, bins: int=4, min_support: float=0.05, alpha: float=0.01, per_column: int=2, quality: bool=True) -> list
```

The most notable findings in `df`, strongest first.

With a `target` (the metric you care about: sales, churn, price …): which segments push it
up or down ("sales is 18% lower when stock < 20"), its trend over time, correlations,
concentration ("20% of regions produce 80% of revenue") and outliers. Segment findings
must pass a Benjamini–Hochberg-corrected significance test (`alpha`) and cover at least
`min_support` of rows. A fitted `model` adds its strongest drivers (SHAP).

---

## `mlpilot.insights.narrate`

<sub>Source: [`mlpilot/insights/narrate.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/insights/narrate.py)</sub>

Turning insights into an executive narrative: an LLM (Claude), or a deterministic template.

Only the insight sentences and their aggregate numbers are sent to the model — never raw
rows. The prompt forbids numbers that aren't in the facts. Without credentials, or on any
API error or refusal, the template narrative is used instead.

### `template_narrative()`

```python
def template_narrative(insights: list, context: dict | None=None) -> str
```

A plain summary built from the insights themselves (no LLM).

### `narrate()`

```python
def narrate(insights: list, *, context: dict | None=None, model: str=DEFAULT_MODEL, client=None, max_tokens: int=4000, effort: str='medium') -> tuple
```

Write the executive summary with Claude. Returns `(text, source)` where `source` is
`"llm:<model>"` or `"template"` (used when the API can't be reached or declines).

`client`: an `anthropic.Anthropic` client (default: one built from the environment —
`ANTHROPIC_API_KEY` or an `ant auth login` profile).

---

## `mlpilot.insights.report`

<sub>Source: [`mlpilot/insights/report.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/insights/report.py)</sub>

Executive insight reports: KPIs, ranked findings and a narrative, as Markdown / HTML / JSON.

### class `InsightsReport`

```python
class InsightsReport:
```

| Field | Type | Default |
|---|---|---|
| `title` | `str` | required |
| `generated` | `str` | required |
| `target` | `str \| None` | required |
| `kpis` | `list` | required |
| `insights` | `list` | required |
| `narrative` | `str` | required |
| `narrative_source` | `str` | required |

#### `InsightsReport.to_dict()`

```python
to_dict() -> dict
```

#### `InsightsReport.to_json()`

```python
to_json(indent: int=2) -> str
```

#### `InsightsReport.to_markdown()`

```python
to_markdown() -> str
```

#### `InsightsReport.to_html()`

```python
to_html() -> str
```

#### `InsightsReport.save()`

```python
save(path) -> Path
```

### `insights_report()`

```python
def insights_report(df, target: str | None=None, *, title: str | None=None, time_col='auto', model=None, llm: bool=False, llm_model: str=DEFAULT_MODEL, client=None, max_insights: int=12, **find_kw) -> InsightsReport
```

An executive report: headline KPIs, ranked plain-English findings and a narrative.

`llm=True` has Claude (`llm_model`, default Claude Opus 5.5) write the narrative from the
findings — only the findings and their aggregate numbers are sent, never raw rows; it falls
back to a template summary without credentials. `llm=False` (default) makes no network calls.
