"""Tests for automatic insights and executive reports."""

import datetime as dt
import json
import types
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import mlpilot as ac
from mlpilot.insights import narrate, template_narrative

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def sales():
    rng = np.random.default_rng(0)
    n = 2000
    days = [dt.date(2024, 1, 1) + dt.timedelta(days=int(d)) for d in rng.integers(0, 365, n)]
    stock = rng.integers(0, 100, n)
    region = rng.choice(["north", "south", "east", "west", "central"], n, p=[0.5, 0.2, 0.1, 0.1, 0.1])
    store = rng.choice([f"s{i}" for i in range(20)], n)
    day_idx = np.array([(d - dt.date(2024, 1, 1)).days for d in days])
    sales = 100 + 0.15 * day_idx - np.where(stock < 20, 30, 0) + rng.normal(0, 10, n)
    return pl.DataFrame({"date": days, "stock": stock, "region": region, "store": store,
                         "noise": rng.normal(size=n), "sales": sales})


def test_drivers_trend_and_phrasing(sales):
    found = ac.find_insights(sales, "sales", max_insights=20)
    heads = [i.headline for i in found]
    stock = next(i for i in found if i.kind == "driver" and i.columns[0] == "stock")
    assert "lower when stock < " in stock.headline and stock.evidence["change"] < -0.1
    assert stock.evidence["q_value"] < 0.01
    assert any(i.kind == "trend" and "rose" in i.headline for i in found)
    assert not any("noise" in h for h in heads)                    # pure noise is not a finding
    assert all(i.columns[0] != "date" for i in found if i.kind == "driver")
    assert found == sorted(found, key=lambda i: -i.strength)


def test_binary_target_rates_and_no_mirror_duplicates():
    rng = np.random.default_rng(1)
    n = 1500
    plan = rng.choice(["monthly", "annual"], n)
    churn = (rng.random(n) < np.where(plan == "monthly", 0.4, 0.1)).astype(int)
    df = pl.DataFrame({"plan": plan, "tenure": rng.integers(1, 60, n), "churn": churn})
    found = ac.find_insights(df, "churn")
    plan_findings = [i for i in found if i.kind == "driver" and i.columns[0] == "plan"]
    assert len(plan_findings) == 1                                 # not both "monthly" and "annual"
    assert "rate of churn = 1" in plan_findings[0].headline


def test_concentration_ignores_missing_group():
    rng = np.random.default_rng(2)
    df = pl.DataFrame({"customer": ["big"] * 5 + [f"c{i}" for i in range(20)] + [None] * 50,
                       "revenue": np.concatenate([rng.uniform(900, 1100, 5), rng.uniform(1, 2, 20),
                                                  rng.uniform(400, 600, 50)])})
    conc = [i for i in ac.find_insights(df, "revenue", quality=False) if i.kind == "concentration"]
    assert conc and "None" not in conc[0].headline and conc[0].evidence["leaders"][0] == "big"


def test_report_template_and_rendering(tmp_path, sales):
    rep = ac.insights_report(sales, "sales", title="Sales review")
    assert rep.narrative_source == "template" and "stock" in rep.narrative
    kpis = dict(rep.kpis)
    assert kpis["Rows"] == "2,000" and "Total sales" in kpis and "Period" in kpis
    html_text = rep.save(tmp_path / "r.html").read_text(encoding="utf-8")
    assert "<title>Sales review · Insights report</title>" in html_text and "Key drivers" in html_text
    md = rep.save(tmp_path / "r.md").read_text(encoding="utf-8")
    assert md.startswith("# Sales review") and "## Evidence" in md
    assert json.loads(rep.to_json())["insights"]


class _FakeClient:
    def __init__(self, text="**Sales rise over the year.**\n\n- Low stock is associated with lower sales.",
                 stop="end_turn", fail=False):
        self.calls, self.text, self.stop, self.fail = [], text, stop, fail
        self.beta = types.SimpleNamespace(messages=types.SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        if self.fail:
            raise ConnectionError("offline")
        block = types.SimpleNamespace(type="text", text=self.text)
        return types.SimpleNamespace(content=[block], stop_reason=self.stop, model=kw["model"])


def test_llm_narrative_request_and_fallbacks(sales):
    found = ac.find_insights(sales, "sales")
    client = _FakeClient()
    text, source = narrate(found, context={"rows": 2000}, client=client)
    assert source == "llm:claude-opus-5-5" and text.startswith("**Sales")
    kw = client.calls[0]
    assert kw["model"] == "claude-opus-5-5" and kw["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in kw["betas"]
    prompt = kw["messages"][0]["content"]
    assert "lower when stock" in prompt and "s17" not in prompt          # findings only, no raw rows
    assert "ONLY the facts" in kw["system"]

    for bad in (_FakeClient(stop="refusal"), _FakeClient(fail=True), _FakeClient(text="")):
        with pytest.warns(RuntimeWarning, match="template"):
            text, source = narrate(found, client=bad)
        assert source == "template" and text == template_narrative(found)

    rep = ac.insights_report(sales, "sales", llm=True, client=_FakeClient())
    assert rep.narrative_source.startswith("llm:") and "written by claude-opus-5-5" in rep.to_markdown()


def test_titanic_with_model():
    path = ROOT / "titanic.csv"
    if not path.exists():
        pytest.skip("titanic.csv not available")
    df = pl.read_csv(path, infer_schema_length=10000)
    model = ac.AutoML("Survived", time_budget=20, cv=3, verbose=False, max_trials=2,
                      include=["logistic_regression", "random_forest"]).fit(df)
    rep = ac.insights_report(df, "Survived", model=model)
    heads = " ".join(i.headline for i in rep.insights)
    assert "Sex" in heads and any(i.kind == "model" for i in rep.insights)
