"""Tests for the shared pipeline core (Step, Pipeline, registry, reports, persistence)."""

import json
import re
from pathlib import Path

import polars as pl
import pytest

import mlpilot as ac
from mlpilot.core import FORMAT


@pytest.fixture
def train():
    return pl.DataFrame({
        "x": [1.0, 2.0, 3.0, None, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "c": ["a", "a", "b", None, "a", "b", "a", "a", "b", "a"],
    })


# ── fit / transform semantics ────────────────────────────────────────────────

def test_transform_reuses_training_statistics(train):
    pipe = ac.Pipeline([ac.Impute(numeric="median", categorical="mode")]).fit(train)
    out = pipe.transform(pl.DataFrame({"x": [None, 100.0, 200.0], "c": [None, "z", "z"]}))
    assert out["x"].to_list() == [6.0, 100.0, 200.0]      # training median, not the batch's
    assert out["c"].to_list() == ["a", "z", "z"]          # training mode


def test_not_fitted_errors(train):
    with pytest.raises(ac.NotFittedError):
        ac.Pipeline([ac.Impute()]).transform(train)
    with pytest.raises(ac.NotFittedError):
        ac.Impute().transform(train)


def test_fit_only_steps_run_on_fit_and_skip_on_transform():
    df = pl.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    pipe = ac.Pipeline([ac.DropDuplicates()])
    assert pipe.fit_transform(df).height == 2
    assert pipe.transform(df).height == 3
    assert pipe.report_.steps[0].skipped is True


def test_pandas_input_is_accepted(train):
    out = ac.Pipeline([ac.Impute()]).fit_transform(train.to_pandas())
    assert isinstance(out, pl.DataFrame) and out["x"].null_count() == 0


def test_step_failure_names_the_step():
    class Boom(ac.Step):
        name = "boom_test"

        def _transform(self, df):
            raise RuntimeError("kaboom")

    with pytest.raises(ac.PipelineStepError, match=r"Step 1 'boom_test' failed during fit: kaboom"):
        ac.Pipeline([Boom()]).fit(pl.DataFrame({"a": [1]}))


# ── building from names and configs ──────────────────────────────────────────

def test_function_steps_resolve_by_name(train):
    pipe = ac.Pipeline(["normalize_whitespace"])
    out = pipe.fit_transform(pl.DataFrame({"s": ["  a   b "]}))
    assert out["s"].to_list() == ["a b"]
    assert isinstance(pipe.steps[0], ac.FunctionStep)


def test_bad_function_params_fail_early():
    with pytest.raises(TypeError, match="normalize_whitespace"):
        ac.Pipeline([{"step": "normalize_whitespace", "params": {"bogus": 1}}])


def test_bad_class_params_fail_early():
    with pytest.raises(TypeError, match="impute"):
        ac.Pipeline([{"step": "impute", "params": {"bogus": 1}}])


def test_unknown_step_suggests_close_names():
    with pytest.raises(ValueError, match="Did you mean: impute"):
        ac.Pipeline(["imputee"])


def test_interactive_functions_are_not_steps():
    for name in ("clean_data", "fix_spelling_errors_in_categorical", "generate_dashboard"):
        with pytest.raises(ValueError, match="Unknown step"):
            ac.Pipeline([name])


def test_fn_prefix_forces_the_function_version(tmp_path, train):
    pipe = ac.Pipeline(["fn:group_rare_categories"])
    assert isinstance(pipe.steps[0], ac.FunctionStep)
    pipe.fit(train)
    reloaded = ac.Pipeline.load(pipe.save(tmp_path / "p.json"))
    assert isinstance(reloaded.steps[0], ac.FunctionStep)
    assert reloaded.to_config()["steps"][0]["step"] == "fn:group_rare_categories"


def test_from_config_dict_json_and_yaml_agree(tmp_path):
    config = {"name": "demo", "steps": [
        {"step": "replace_disguised_missing"},
        {"step": "impute", "params": {"numeric": "mean"}},
        {"step": "drop_duplicates", "fit_only": False},
    ]}
    a = ac.Pipeline.from_config(config)
    a.save_config(tmp_path / "c.json")
    a.save_config(tmp_path / "c.yaml")
    b = ac.Pipeline.from_config(tmp_path / "c.json")
    c = ac.Pipeline.from_config(tmp_path / "c.yaml")
    assert a.config_hash() == b.config_hash() == c.config_hash()
    assert c.steps[1].params["numeric"] == "mean"
    assert c.steps[2].fit_only is False


def test_available_steps_lists_stateful_and_function_steps():
    steps = ac.available_steps()
    assert steps["impute"]["kind"] == "stateful"
    assert steps["normalize_unicode"]["kind"] == "function"
    assert steps["fn:parse_dates"]["kind"] == "function"          # shadowed by ParseDates


def test_custom_step_registration_round_trips(tmp_path):
    @ac.register_step
    class AddConstant(ac.Step):
        """Add a learned constant column."""
        name = "add_constant_test"

        def __init__(self, *, value=1, fit_only=None):
            super().__init__(fit_only=fit_only, value=value)

        def _fit(self, df):
            return {"value": self.params["value"] * 2}

        def _transform(self, df):
            return df.with_columns(pl.lit(self.state_["value"]).alias("k"))

    pipe = ac.Pipeline([{"step": "add_constant_test", "params": {"value": 21}}])
    pipe.fit(pl.DataFrame({"a": [1]}))
    out = ac.Pipeline.load(pipe.save(tmp_path / "p.json")).transform(pl.DataFrame({"a": [5]}))
    assert out["k"].to_list() == [42]


# ── persistence ──────────────────────────────────────────────────────────────

def test_save_load_reproduces_transform_exactly(tmp_path, train):
    pipe = ac.default_cleaning_pipeline().fit(train)
    path = pipe.save(tmp_path / "pipe.json")
    new = pl.DataFrame({"x": [None, 4.0, 1000.0], "c": ["b", None, "zzz"]})
    assert ac.Pipeline.load(path).transform(new).equals(pipe.transform(new))


def test_saved_pipeline_is_plain_json(tmp_path, train):
    path = ac.Pipeline([ac.Impute()]).fit(train).save(tmp_path / "pipe.json")
    data = json.loads(Path(path).read_text(encoding="utf-8"))   # JSON, never pickle
    assert data["format"] == FORMAT
    assert data["steps"][0]["state"]["fill"]["x"] == 6.0
    assert data["input_schema"] == {"x": "Float64", "c": "String"}


def test_loading_a_config_file_as_pipeline_is_rejected(tmp_path):
    ac.Pipeline(["impute"]).save_config(tmp_path / "c.json")
    with pytest.raises(ValueError, match="not a saved pipeline"):
        ac.Pipeline.load(tmp_path / "c.json")


# ── schema checks ────────────────────────────────────────────────────────────

def test_schema_mismatch_policies(train):
    batch = train.drop("c")
    with pytest.raises(ac.SchemaMismatchError, match="missing columns"):
        ac.Pipeline([ac.Impute()], on_schema_mismatch="raise").fit(train).transform(batch)

    warn_pipe = ac.Pipeline([ac.Impute()], on_schema_mismatch="warn").fit(train)
    with pytest.warns(ac.PipelineWarning, match="missing columns"):
        warn_pipe.transform(batch)
    assert warn_pipe.report_.warnings

    quiet = ac.Pipeline([ac.Impute()], on_schema_mismatch="ignore").fit(train)
    quiet.transform(batch)
    assert quiet.report_.warnings == []


# ── reports & audit log ──────────────────────────────────────────────────────

def test_run_report_and_audit_log(tmp_path, train):
    log = tmp_path / "audit.jsonl"
    pipe = ac.Pipeline([ac.DropDuplicates(), ac.Impute()], name="demo", audit_log=log)
    pipe.fit_transform(train)
    fit_report = pipe.report_
    assert fit_report.mode == "fit" and len(fit_report.steps) == 2
    assert fit_report.input["rows"] == 10 and fit_report.output["nulls"] == 0
    assert re.fullmatch(r"[0-9a-f]{16}", fit_report.config_hash)
    assert "impute" in fit_report.summary()
    assert fit_report.to_frame().height == 2
    json.loads(fit_report.to_json())

    pipe.transform(train)
    assert pipe.report_.run_id != fit_report.run_id
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(line)["mode"] for line in lines] == ["fit", "transform"]


def test_step_output_is_captured_in_report_not_printed(capsys):
    pipe = ac.Pipeline(["standardize_column_names"])
    pipe.fit(pl.DataFrame({"A B": [1]}))
    assert capsys.readouterr().out == ""
    assert "Standardizing column names" in pipe.report_.steps[0].log


# ── architecture guard ───────────────────────────────────────────────────────

def test_core_does_not_import_the_rest_of_the_package():
    """The core must stay extractable into its own package for future modules."""
    forbidden = re.compile(r"^\s*(from\s+\.\.|from\s+mlpilot|import\s+mlpilot)",
                           re.MULTILINE)
    core_dir = Path(ac.__file__).parent / "core"
    for src in core_dir.glob("*.py"):
        assert not forbidden.search(src.read_text(encoding="utf-8")), src.name
