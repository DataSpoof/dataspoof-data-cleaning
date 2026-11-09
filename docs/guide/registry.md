# Experiment tracking & model registry


Both are plain directories of JSON files. There is no server to run, and they are easy to back
up, diff or keep on a shared drive.

```python
tracker = ac.Tracker("ac_runs", experiment="titanic")
with tracker.start_run("baseline") as run:           # ends "finished", or "failed" with the error
    model = ac.AutoML("Survived").fit(train)
    run.log_automl(model, train, test)               # params, CV + test metrics, dataset fingerprints,
                                                     # leaderboard, the model and its model card
    run.log_metric("custom", 0.9, step=1)            # anything else: log_params / log_artifact / set_tag

tracker.runs()                                       # every run as a table (params, latest metrics)
tracker.best_run("test_roc_auc")
tracker.compare([run_a.id, run_b.id])
```

Each run records the git commit, the user and its timings. A dataset is recorded by its shape,
schema and content hash, never by its rows.

```python
reg = ac.ModelRegistry("ac_registry")
v = reg.register(model, "titanic", run=run)          # -> v1, stage "dev"
reg.promote("titanic", v.version, "staging")
reg.promote("titanic", v.version, "production",
            require={"test_roc_auc": 0.85},          # metric gates; failing ones raise PromotionError
            must_beat_production="test_roc_auc")     # ...and it must beat the current production model
model = reg.load("titanic")                          # the production model (checksum verified)
reg.history("titanic")                               # who moved what, when and why
reg.rollback("titanic")                              # restore the previous production version
```
