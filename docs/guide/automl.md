# AutoML


```python
model = ac.AutoML("churned", time_budget=120).fit(train_df)
print(model.summary())               # task, metric, CV scheme, best model, best score per model
model.leaderboard_                   # every trial with its score, std, time and parameters
model.evaluate(test_df)              # accuracy, balanced accuracy, F1, ROC-AUC, log-loss (RMSE, MAE, R² for regression)
model.predict(new_df)                # labels in their original form
model.predict_proba(new_df)          # one proba_<class> column per class
model.feature_importance()           # permutation importance
model.save("churn_model/")           # directory: manifest.json + model.joblib + preprocess.json
model = ac.AutoML.load("churn_model/")
```

`AutoML(target).fit(df)` works through these steps:

1. **Detect the task.** Two values means binary; text, or up to 10 whole-number values, means
   multiclass; anything else is regression. Rows sorted by a date column are treated as a time
   series.
2. **Pick a metric:** ROC-AUC, macro-F1 or RMSE.
3. **Drop leaky columns.**
4. **Clean and engineer features** with the target-aware pipelines above.
5. **Cross-validate a baseline and every model.** The defaults are logistic/ridge regression,
   histogram gradient boosting, random forest, extra trees, k-NN, and LightGBM and XGBoost if
   installed. Folds are stratified, or time-ordered for time series.
6. **Spend the rest of `time_budget` on random hyperparameter search** around the best model
   families.
7. **Try a soft-voting ensemble** of the top three families.
8. **Refit the winner on all rows.**

| Option | Default | Meaning |
|---|---|---|
| `time_budget` | 60 | seconds for the search (75% models, the rest ensemble + refit) |
| `max_trials` | None | cap on model trials |
| `metric` | auto | any scikit-learn scorer name, e.g. `"average_precision"` |
| `cv` | 5 | number of folds |
| `include` / `exclude` | all | model names, see `ac.available_models("binary")` |
| `preprocess` | `"auto"` | `"auto"`, your own `Pipeline`, or `None` |
| `ensemble` | True | try the voting ensemble |
| `on_leakage` | `"drop"` | `drop`, `warn`, `raise` or `ignore` |
| `time_col` | `"auto"` | a date column for time-ordered CV, or None to switch detection off |

On Titanic, with the defaults and a 40-second budget, AutoML reaches 0.918 ROC-AUC and 83% accuracy
on held-out rows.

**Saved models:** a trained model can't be stored as JSON, so `model.joblib` uses joblib, the
scikit-learn standard. Its SHA-256 is recorded in `manifest.json` and verified before loading,
which catches corrupted or swapped files. Unpickling still runs code, so only load models from
people you trust.

### Uncertainty, imbalance and smarter search

```python
model = ac.AutoML("price", search="optuna").fit(train)   # Bayesian (TPE) search: pip install "...[optuna]"
model.predict_interval(test, coverage=0.9)                # prediction, lower, upper (conformal)
clf.predict_set(test, coverage=0.9)                       # label sets that contain the truth 90% of the time
```

- **Conformal intervals and sets** are calibrated on out-of-fold errors, so the stated
  coverage holds on new data (90% intervals covered 90.2% of held-out values in testing).
- **Imbalanced classes:** with `imbalance="auto"` (the default), training uses balanced class
  weights when the smallest class is under 20%.
- **Parallel cross-validation:** `n_jobs=-1` runs the folds in parallel.
- **Fast scoring:** predictions run single-threaded by default, which keeps single-row API
  latency low. Use `set_inference_threads(-1)` for large batch jobs.

### Clustering

```python
clusters = ac.AutoCluster(k_range=(2, 8)).fit(df)     # k-means and Gaussian mixtures, chosen by silhouette
clusters.labels_                                      # one label per input row
clusters.cluster_profiles(df)                         # size, share and averages per cluster
ac.automl(df)                                         # no target -> AutoCluster; with target= -> AutoML
```

Optional extras: `pip install "AutomatedCleaning[automl]"` (LightGBM, XGBoost) and
`pip install "AutomatedCleaning[embeddings]"` (sentence-transformers).
