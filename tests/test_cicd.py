"""Model CI/CD: the pipeline steps run through the real CLI, and the generated pipelines."""

import json

import numpy as np
import polars as pl
import pytest
import yaml

from mlpilot.deploy.cli import main


@pytest.fixture()
def project(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n = 900
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    df = pl.DataFrame({"customer": rng.integers(0, 300, n), "x1": x1, "x2": x2,
                       "plan": rng.choice(["basic", "pro"], n),
                       "churned": ((1.5 * x1 + x2 + rng.normal(0, 0.5, n)) > 0).astype(int)})
    (tmp_path / "data").mkdir()
    df.write_parquet(tmp_path / "data" / "train.parquet")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_pipeline_end_to_end_through_the_cli(project, capsys):
    assert main(["cicd", "init", "--provider", "github", "--name", "churn", "--data", "data/train.parquet",
                 "--target", "churned", "--floor", "0.75"]) == 0
    wf = project / ".github" / "workflows" / "model-churn.yml"
    assert wf.exists() and (project / "gates.yml").exists() and (project / "environments.yml").exists()
    jobs = yaml.safe_load(wf.read_text())["jobs"]
    assert list(jobs) == ["train", "register", "release-qa", "deploy-production"]
    assert jobs["deploy-production"]["environment"] == "production" and jobs["release-qa"]["needs"] == "register"

    assert main(["train", "--data", "data/train.parquet", "--target", "churned", "--out", "build/model",
                 "--group", "customer", "--time-budget", "20", "--include", "logistic_regression",
                 "--report", "build/train.json"]) == 0
    rep = json.loads((project / "build" / "train.json").read_text())
    assert rep["split"].startswith("grouped") and rep["split_issues"] == [] and rep["metrics"]["test_roc_auc"] > 0.8
    assert (project / "build" / "model_card.md").exists() and (project / "build" / "test.parquet").exists()

    assert main(["gate", "--model", "build/model", "--data", "build/test.parquet", "--gates", "gates.yml",
                 "--report", "build/gate.json"]) == 0
    gate = json.loads((project / "build" / "gate.json").read_text())
    assert gate["passed"] and "robustness" in gate["checks"]
    (project / "strict.yml").write_text("metrics: {min: {roc_auc: 0.999}}\nrobustness: {max_noise_flip: 0.0}\n")
    capsys.readouterr()
    assert main(["gate", "--model", "build/model", "--data", "build/test.parquet", "--gates", "strict.yml"]) == 1
    out = capsys.readouterr().out
    assert "GATES FAILED" in out and "roc_auc" in out

    assert main(["register", "--model", "build/model", "--registry", "registry/dev", "--name", "churn",
                 "--metrics", "build/metrics.json", "--stage", "staging", "--report", "build/register.json"]) == 0
    version = json.loads((project / "build" / "register.json").read_text())["version"]
    assert main(["release", "--config", "environments.yml", "--name", "churn", "--version", str(version),
                 "--from", "dev", "--to", "qa", "--released-by", "ci"]) == 0
    capsys.readouterr()
    assert main(["--registry", "registry/qa", "latest", "--name", "churn", "--stage", "qa"]) == 0
    qa_version = int(capsys.readouterr().out.strip())
    assert main(["release", "--config", "environments.yml", "--name", "churn", "--version", str(qa_version),
                 "--from", "qa", "--to", "prod", "--released-by", "ci", "--report", "build/release.json"]) == 0
    assert json.loads((project / "build" / "release.json").read_text())["stage"] == "staging"
    capsys.readouterr()
    # skipping QA is refused with a clear message and exit code 1
    assert main(["release", "--config", "environments.yml", "--name", "churn", "--version", str(version),
                 "--from", "dev", "--to", "prod", "--released-by", "ci"]) == 1
    assert "can't be skipped" in capsys.readouterr().err


def test_gitlab_pipeline_and_files_are_not_overwritten(project):
    assert main(["cicd", "init", "--provider", "gitlab", "--name", "churn", "--data", "data/train.parquet",
                 "--target", "churned"]) == 0
    ci = yaml.safe_load((project / ".gitlab-ci.yml").read_text())
    assert ci["stages"] == ["train", "gate", "register", "release-qa", "deploy"]
    assert ci["deploy-production"]["rules"][0]["when"] == "manual" and ci["gate"]["needs"] == ["train"]
    (project / "gates.yml").write_text("# mine\n")
    main(["cicd", "init", "--provider", "gitlab", "--name", "churn", "--data", "data/train.parquet", "--target", "churned"])
    assert (project / "gates.yml").read_text() == "# mine\n"


def test_poisoned_training_data_fails_the_build(project):
    df = pl.read_parquet(project / "data" / "train.parquet")
    bad = df.with_columns(pl.when(pl.int_range(pl.len()) % 6 == 0).then(1 - pl.col("churned"))
                          .otherwise(pl.col("churned")).alias("churned"))
    bad.write_parquet(project / "data" / "bad.parquet")
    with pytest.raises(SystemExit, match="poisoned or mislabelled"):
        main(["train", "--data", "data/bad.parquet", "--target", "churned", "--out", "build/m2", "--time-budget", "10",
              "--include", "logistic_regression", "--max-suspicious-share", "0.05"])
