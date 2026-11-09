"""Tests for the profiling & data-quality module."""

import json
from pathlib import Path

import polars as pl
import pytest

import mlpilot as ac
from mlpilot.quality import semantic_type
from mlpilot.quality.checks import _case_variant_cells

N = 120


@pytest.fixture
def messy():
    adv = [str(i) for i in range(N)]
    adv[3] = adv[7] = "-"
    signup = ["2021-01-%02d" % (i % 28 + 1) if i % 3 else "05/%02d/2021" % (i % 12 + 1) for i in range(N)]
    signup[5] = "2021-13-40"
    status = ["active" if i % 2 else "inactive" for i in range(N)]
    status[:3] = ["NA", "?", "unknown"]
    income = [float(30000 + i * 100) for i in range(N)]
    income[:5] = [-999.0] * 5
    age = [float(20 + i % 50) for i in range(N)]
    age[:2] = [150.0, -3.0]
    price = [float(10 + i) for i in range(N)]
    price[0] = -5.0
    email = [f"user{i}@mail.com" for i in range(N)]
    email[:2] = ["bad-email", "x@y"]
    phone = ["+1 415-555-%04d" % i for i in range(N)]
    phone[:2] = ["12345", "call me"]
    return pl.DataFrame({
        "adv": adv, "price_txt": [f"${i * 10:,}" for i in range(N)], "signup": signup,
        "ship_date": ["2022-03-%02d" % (i % 28 + 1) for i in range(N)], "status": status,
        "income": income, "age": age, "price": price,
        "city": ["Delhi"] * (N - 3) + ["Pune", "Goa", "Agra"], "email": email, "phone": phone,
        "zip": ["0%04d" % i for i in range(N)],
    })


# ── checks ───────────────────────────────────────────────────────────────────

def test_messy_dataset_reports_each_problem_once(messy):
    titles = {i.title for i in ac.check_quality(messy).issues}
    expected = {
        "Disguised missing values · adv", "Disguised missing values · status",
        "Placeholder numbers · income", "Numbers stored as text · adv",
        "Numbers stored as text · price_txt", "Mixed date formats · signup",
        "Dates stored as text · ship_date", "Out-of-range values · age",
        "Out-of-range values · price", "Rare categories · city",
        "Invalid email addresses · email", "Invalid phone numbers · phone",
    }
    assert expected <= titles
    false_positives = {
        "Possible identifier · price", "Possible identifier · income",  # continuous floats
        "Possible identifier · adv",                                     # numbers stored as text
        "Outliers · income", "Out-of-range values · income",             # -999 is a placeholder
        "Skewed distribution · income", "Rare categories · status",      # placeholders, not categories
    }
    assert not titles & false_positives


def test_clean_data_scores_100():
    df = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["x", "y", "x", "y"]})
    report = ac.check_quality(df)
    assert report.score == 100.0 and report.grade == "A"
    assert all(v == 100.0 for v in report.dimensions.values())


def test_completeness_counts_text_placeholders():
    report = ac.check_quality(pl.DataFrame({"s": ["a", "NA", "b", "?"]}))
    assert report.dimensions["completeness"] == 50.0       # no nulls, but 2 of 4 are placeholders


def test_uniqueness_counts_extra_copies():
    report = ac.check_quality(pl.DataFrame({"a": [1, 1, 2, 3]}))
    assert report.dimensions["uniqueness"] == 75.0
    assert report.filter(check="duplicate_rows")[0].affected == 1


def test_validity_and_consistency():
    df = pl.DataFrame({"age": [10.0, 20.0, 30.0, 200.0],
                       "city": ["Delhi", "Delhi", "delhi", " Goa"]})
    report = ac.check_quality(df)
    assert report.dimensions["validity"] == 75.0           # 1 of 4 ages out of [0, 120]
    assert report.dimensions["consistency"] == 50.0        # 'delhi' variant + ' Goa' whitespace


def test_case_variant_cells():
    assert _case_variant_cells(pl.Series(["USA", "USA", "usa", "Usa", "India"])) == 2


def test_zero_width_iqr_is_not_reported_as_outliers():
    df = pl.DataFrame({"parch": [0] * 90 + [1, 2, 3, 4, 5, 6, 1, 2, 1, 2]})
    assert not ac.check_quality(df).filter(check="outliers")


def test_custom_weights_and_validation():
    df = pl.DataFrame({"a": [1, None]})
    only_completeness = ac.check_quality(df, weights={"completeness": 1})
    assert only_completeness.score == 50.0
    with pytest.raises(ValueError):
        ac.check_quality(df, weights={"freshness": 1})


def test_report_exports_and_weakest_columns(messy):
    report = ac.check_quality(messy)
    frame = report.to_frame()
    assert frame.height == len(report.issues) and "severity" in frame.columns
    data = json.loads(report.to_json())
    assert data["score"] == report.score and data["counts"] == report.counts


def test_weakest_columns_expose_a_broken_column_the_overall_score_hides():
    df = pl.DataFrame({f"c{k}": [float(i * (k + 1)) for i in range(20)] for k in range(9)})
    df = df.with_columns(pl.Series("broken", [None] * 16 + [1, 2, 3, 4]))
    report = ac.check_quality(df)
    assert report.score >= 90                                 # 92% of cells are fine ...
    assert report.weakest_columns() == [("broken", 20.0)]     # ... but this column is 80% empty
    assert "Weakest columns: broken 20.0" in report.summary()


# ── profile ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("values,expected", [
    (list(range(20)), "identifier"),
    ([0, 1] * 10, "binary"),
    ([1, 2, 3] * 7, "discrete"),
    ([1.5, 2.25, 3.75, 9.5, 10.0] * 3, "numeric"),
    (["1,200", "300", "45", "-", "7"] * 2, "numeric_text"),
    (["2021-01-01", "2021-02-03", "2021-03-04"] * 3, "datetime_text"),
    (["a@x.com", "b@y.org", "c@z.io"] * 3, "email"),
    (["https://a.com", "http://b.org/x", "https://c.io"] * 3, "url"),
    (["red", "green", "blue"] * 5, "categorical"),
    (["x"] * 5, "constant"),
    ([None, None], "empty"),
    ([True, False, True], "boolean"),
])
def test_semantic_types(values, expected):
    assert semantic_type(pl.Series("col", values)) == expected


def test_phone_semantic_type_by_name():
    assert semantic_type(pl.Series("mobile", ["+1 415 555 0100", "+1 415 555 0101"])) == "phone"


def test_profile_statistics_correlations_and_insights():
    df = pl.DataFrame({
        "id": list(range(50)),
        "x": [float(i) for i in range(50)],
        "y": [float(i) * 2 + (i % 3) for i in range(50)],
        "empty_ish": [None] * 30 + ["a"] * 20,
        "cat": ["a"] * 45 + ["b"] * 5,
    })
    p = ac.profile(df)
    assert p["id"].semantic_type == "identifier"
    assert p["x"].numeric["median"] == 24.5 and p["x"].numeric["min"] == 0.0
    assert p["empty_ish"].null_pct == 60.0 and p["cat"].text["max_length"] == 1
    assert p.correlations[0]["a"] == "x" and p.correlations[0]["b"] == "y"
    joined = " ".join(p.insights)
    assert "“empty_ish” is 60% empty" in joined and "correlated" in joined
    assert "dominated by 'a'" in joined
    assert p.to_frame().height == 5
    assert json.loads(p.to_json())["rows"] == 50
    assert p.quality is not None and ac.profile(df, quality=False).quality is None


# ── contracts ────────────────────────────────────────────────────────────────

CONTRACT_YAML = """
name: customers
extra_columns: error
no_duplicate_rows: true
unique: [customer_id]
row_count: {min: 2, max: 100}
columns:
  customer_id: {type: integer, required: true, unique: true}
  email:       {type: string, format: email}
  age:         {type: number, min: 0, max: 120, max_null_fraction: 0.25}
  plan:        {type: string, allowed: [basic, pro]}
  code:        {type: string, pattern: '^[A-Z]{3}$', min_length: 3, max_length: 3}
  phone:       {type: string, format: phone, severity: warn}
"""


def _good():
    return pl.DataFrame({
        "customer_id": [1, 2, 3, 4], "email": ["a@x.com", "b@x.com", "c@x.com", "d@x.com"],
        "age": [30.0, 40.0, None, 50.0], "plan": ["basic", "pro", "pro", "basic"],
        "code": ["ABC", "DEF", "GHI", "JKL"],
        "phone": ["+1 415 555 0100", "+1 415 555 0101", "+1 415 555 0102", "+1 415 555 0103"],
    })


def test_contract_passes_on_good_data_and_round_trips(tmp_path):
    contract = ac.Contract.from_yaml(CONTRACT_YAML)
    result = contract.validate(_good())
    assert result.passed and not result.failures, result.summary()
    contract.to_yaml(tmp_path / "c.yaml")
    contract.to_json(tmp_path / "c.json")
    for path in ("c.yaml", "c.json"):
        assert ac.Contract.from_file(tmp_path / path).to_dict() == contract.to_dict()


def test_contract_catches_every_rule():
    bad = pl.DataFrame({
        "customer_id": [1, 1, None, 4], "email": ["a@x.com", "nope", "c@x.com", "d@x.com"],
        "age": ["30", "40", "50", "60"],                          # wrong type
        "plan": ["basic", "gold", "pro", "basic"],
        "code": ["ABC", "de", "GHI", "JKLM"],
        "phone": ["+1 415 555 0100", "123", "+1 415 555 0102", "+1 415 555 0103"],
        "extra": [1, 2, 3, 4],
    })
    result = ac.Contract.from_yaml(CONTRACT_YAML).validate(bad)
    failed = {(r.column, r.rule) for r in result.failures}
    assert {("customer_id", "required"), ("customer_id", "unique"), ("customer_id", "unique_key"),
            ("email", "format"), ("age", "type"), ("age", "min"), ("plan", "allowed"),
            ("code", "pattern"), ("code", "min_length"), ("code", "max_length"),
            ("phone", "format"), (None, "extra_columns")} <= failed
    assert not result.passed
    phone = next(r for r in result.failures if r.column == "phone")
    assert phone.severity == "warn" and phone.examples == ["123"]
    with pytest.raises(ac.ContractViolation) as excinfo:
        result.raise_if_failed()
    assert excinfo.value.result is result


def test_missing_column_row_count_and_duplicates():
    df = pl.DataFrame({"customer_id": [1, 1]})
    contract = ac.Contract({"customer_id": {"type": "integer"}, "email": {"type": "string"}},
                           row_count={"min": 5}, no_duplicate_rows=True)
    failed = {(r.column, r.rule) for r in contract.validate(df).failures}
    assert {("email", "column_present"), (None, "row_count"), (None, "no_duplicate_rows")} <= failed


def test_warn_only_rules_do_not_fail_the_contract():
    contract = ac.Contract({"age": {"type": "number", "max": 100, "warn_only": ["max"]}})
    result = contract.validate(pl.DataFrame({"age": [10.0, 150.0]}))
    assert result.passed and len(result.warnings) == 1 and result.warnings[0].examples == [150.0]


def test_contract_definition_errors_are_helpful():
    with pytest.raises(ac.ContractError, match="Did you mean 'allowed'"):
        ac.Contract({"plan": {"alowed": ["a"]}})
    with pytest.raises(ac.ContractError, match="type must be one of"):
        ac.Contract({"x": {"type": "float64"}})
    with pytest.raises(ac.ContractError, match="format must be one of"):
        ac.Contract({"x": {"format": "iban"}})
    with pytest.raises(ac.ContractError, match="Did you mean 'columns'"):
        ac.Contract.from_dict({"colums": {}})


def test_inferred_contract_accepts_source_and_flags_drift():
    source = pl.DataFrame({
        "id": list(range(20)), "plan": ["basic", "pro"] * 10,
        "spend": [float(i) for i in range(20)], "email": [f"u{i}@x.com" for i in range(20)],
    })
    contract = ac.Contract.infer(source)
    spec = contract.columns
    assert spec["id"] == {"type": "integer", "required": True, "unique": True,
                          "min": 0, "max": 19, "warn_only": ["min", "max"]}
    assert spec["plan"]["allowed"] == ["basic", "pro"] and spec["email"]["format"] == "email"
    assert contract.validate(source).passed and not contract.validate(source).failures

    drifted = source.with_columns(pl.col("spend") * 10, pl.lit("gold").alias("plan"))
    result = contract.validate(drifted)
    assert result.passed                                      # statistical drift only warns ...
    assert {(r.column, r.rule) for r in result.warnings} >= {("spend", "max"), ("plan", "allowed")}
    broken = source.with_columns(pl.col("id").cast(pl.String))
    assert not contract.validate(broken).passed               # ... a type change is an error


# ── schema ───────────────────────────────────────────────────────────────────

def test_compare_schemas():
    a = pl.DataFrame({"x": [1], "y": ["a"]})
    b = pl.DataFrame({"x": [1.0], "z": [True]})
    diff = ac.compare_schemas(a, b)
    assert diff.removed == ["y"] and diff.added == ["z"] and diff.changed == {"x": ["Int64", "Float64"]}
    assert not diff.compatible and ac.compare_schemas(a, a).identical
    assert ac.infer_schema(a) == {"x": "Int64", "y": "String"}


# ── pipeline steps ───────────────────────────────────────────────────────────

def test_validate_contract_step_infers_saves_and_blocks(tmp_path):
    train = pl.DataFrame({"id": list(range(20)), "plan": ["basic", "pro"] * 10})
    pipe = ac.Pipeline([ac.ValidateContract()]).fit(train)
    reloaded = ac.Pipeline.load(pipe.save(tmp_path / "p.json"))      # contract travels with it
    assert reloaded.transform(train).equals(train)
    retyped = train.with_columns(pl.col("id").cast(pl.String))
    with pytest.warns(ac.PipelineWarning, match="dtype changes"):    # the core's schema check
        with pytest.raises(ac.PipelineStepError) as excinfo:          # ... and the contract
            reloaded.transform(retyped)
    assert isinstance(excinfo.value.__cause__, ac.ContractViolation)


def test_validate_contract_step_with_explicit_contract_and_warn_mode(tmp_path):
    path = tmp_path / "c.yaml"
    ac.Contract({"a": {"type": "integer", "max": 5}}).to_yaml(path)
    step = ac.ValidateContract(contract=str(path), on_fail="warn")
    pipe = ac.Pipeline([step]).fit(pl.DataFrame({"a": [1, 2]}))
    with pytest.warns(ac.QualityWarning, match="max"):
        pipe.transform(pl.DataFrame({"a": [1, 9]}))
    assert not step.last_result_.passed


def test_quality_gate():
    bad = pl.DataFrame({"a": [None, None, None, 1.0]})
    with pytest.raises(ac.PipelineStepError) as excinfo:
        ac.Pipeline([ac.QualityGate(min_score=90)]).fit(bad)
    assert isinstance(excinfo.value.__cause__, ac.QualityGateError)
    gate = ac.QualityGate(min_score=None, min_dimensions={"completeness": 50}, on_fail="warn")
    with pytest.warns(ac.QualityWarning, match="completeness"):
        ac.Pipeline([gate]).fit(bad)
    assert gate.last_report_.dimensions["completeness"] == 25.0
    with pytest.raises(ValueError):
        ac.QualityGate(min_dimensions={"freshness": 1})


# ── dashboard integration ────────────────────────────────────────────────────

def test_dashboard_renders_score_and_issues_from_quality_module(tmp_path, monkeypatch, messy):
    pytest.importorskip("plotly")           # the dashboard needs the [dashboard] extra
    monkeypatch.chdir(tmp_path)
    df = messy.with_columns(pl.Series("notes", [None] * 100 + ["x"] * 20))   # a weak column
    ac.generate_dashboard(df)
    html = Path("output/eda/dashboard.html").read_text(encoding="utf-8")
    report = ac.check_quality(df)
    assert "Data quality score" in html and f"{report.score:.1f}" in html
    assert "Mixed date formats · signup" in html and "Weakest columns" in html
