"""Executive insight reports: KPIs, ranked findings and a narrative, as Markdown / HTML / JSON."""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
from pathlib import Path

import polars as pl

from .._detect import is_num, is_temporal
from ..quality.checks import jsonable
from .finder import Insight, _num, _positive, _target_kind, find_insights
from .narrate import DEFAULT_MODEL, narrate, template_narrative

_SECTIONS = (("Key drivers", ("driver", "model")), ("Trends", ("trend", "change")),
             ("Patterns", ("correlation", "concentration", "outliers")), ("Data notes", ("quality",)))


@dataclasses.dataclass
class InsightsReport:
    title: str
    generated: str
    target: str | None
    kpis: list                    # [(label, value)]
    insights: list
    narrative: str
    narrative_source: str         # "template" | "llm:<model>"

    def to_dict(self) -> dict:
        return jsonable({"title": self.title, "generated": self.generated, "target": self.target,
                         "kpis": [{"label": k, "value": v} for k, v in self.kpis],
                         "insights": [dataclasses.asdict(i) for i in self.insights],
                         "narrative": self.narrative, "narrative_source": self.narrative_source})

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def _blocks(self) -> list:
        source = ("written by " + self.narrative_source[4:] + " from the findings below"
                  if self.narrative_source.startswith("llm:") else "generated from the findings below")
        b: list[tuple] = [("p", f"Generated {self.generated} by MLPilot")]
        b.append(("h2", "At a glance"))
        b.append(("kv", self.kpis))
        b.append(("h2", "Summary"))
        para = []
        for line in self.narrative.splitlines():
            s = line.strip()
            if s.startswith(("- ", "* ")):
                para.append(s[2:])
                continue
            if para:
                b.append(("list", para))
                para = []
            if s.startswith("#"):
                b.append(("h3", s.lstrip("# ")))
            elif s:
                b.append(("p", s))
        if para:
            b.append(("list", para))
        caveat = (" Findings are associations in historical data, not proven causes."
                  if self.narrative_source.startswith("llm:") else "")
        b.append(("p", f"_Summary {source}.{caveat}_"))
        for title, kinds in _SECTIONS:
            items = [i for i in self.insights if i.kind in kinds]
            if items:
                b.append(("h2", title))
                b.append(("list", [i.headline for i in items]))
        drivers = [i for i in self.insights if i.kind == "driver"]
        if drivers:
            b.append(("h2", "Evidence"))
            rows = []
            for i in drivers:
                e = i.evidence
                fmt = (lambda v: f"{v:.1%}") if e["measure"] == "rate" else _num
                rows.append([e["segment"], f"{e['size']:,}", fmt(e["value"]), fmt(e["rest"]),
                             f"{e['change']:+.0%}" if e["measure"] == "average" else f"{e['change'] * 100:+.1f} pts"])
            b.append(("table", ["Segment", "Rows", "In segment", "Everyone else", "Difference"], rows))
        return b

    def to_markdown(self) -> str:
        from ..evaluation.card import _render_markdown
        return f"# {self.title}\n\n" + _render_markdown(self._blocks())

    def to_html(self) -> str:
        from ..evaluation.card import _render_html
        return _render_html(self.title, self._blocks(), doc_kind="Insights report")

    def save(self, path) -> Path:
        path = Path(path)
        writers = {".md": self.to_markdown, ".html": self.to_html, ".htm": self.to_html, ".json": self.to_json}
        if path.suffix.lower() not in writers:
            raise ValueError("Save an insights report as .md, .html or .json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(writers[path.suffix.lower()](), encoding="utf-8")
        return path

    def _repr_html_(self):
        return self.to_html()

    def __repr__(self) -> str:
        return f"{self.title}\n\n{self.narrative}"


def _kpis(df: pl.DataFrame, target, time_col) -> list:
    k = [("Rows", f"{df.height:,}"), ("Columns", str(df.width))]
    if time_col:
        k.append(("Period", f"{df[time_col].min()} to {df[time_col].max()}"))
    if target:
        kind = _target_kind(df[target])
        if kind == "numeric":
            s = df[target].cast(pl.Float64)
            k += [(f"Total {target}", _num(s.sum())), (f"Average {target}", _num(s.mean())),
                  (f"Median {target}", _num(s.median()))]
        elif kind in ("binary", "categorical"):
            pos = _positive(df[target]) if kind == "binary" else df[target].drop_nulls().mode().to_list()[0]
            rate = float((df[target] == pos).fill_null(False).mean())
            k.append((f"{target} = {pos}", f"{rate:.1%} of rows"))
        missing = df[target].null_count()
        if missing:
            k.append((f"{target} missing", f"{missing:,} rows"))
    return k


def insights_report(df, target: str | None = None, *, title: str | None = None, time_col="auto", model=None,
                    llm: bool = False, llm_model: str = DEFAULT_MODEL, client=None, max_insights: int = 12,
                    **find_kw) -> InsightsReport:
    """An executive report: headline KPIs, ranked plain-English findings and a narrative.

    ``llm=True`` has Claude (``llm_model``, default Claude Opus 5.5) write the narrative from the
    findings — only the findings and their aggregate numbers are sent, never raw rows; it falls
    back to a template summary without credentials. ``llm=False`` (default) makes no network calls.
    """
    df = df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
    if time_col == "auto":
        time_col = next((c for c, t in df.schema.items() if is_temporal(t) and df[c].n_unique() > 2), None)
    insights = find_insights(df, target, time_col=time_col, model=model, max_insights=max_insights, **find_kw)
    context = {"rows": df.height, "columns": df.width, "target": target, "time_column": time_col}
    if llm:
        text, source = narrate(insights, context=context, model=llm_model, client=client)
    else:
        text, source = template_narrative(insights, context), "template"
    return InsightsReport(title=title or (f"What drives {target}" if target else "Data insights"),
                          generated=_dt.datetime.now().isoformat(timespec="seconds"), target=target,
                          kpis=_kpis(df, target, time_col), insights=insights, narrative=text,
                          narrative_source=source)
