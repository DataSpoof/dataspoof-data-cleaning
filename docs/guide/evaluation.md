# Evaluation & explainability


Every tool takes a fitted `AutoML` model (or any scikit-learn estimator plus `target=...`) and
labelled data. Each is also a method on `AutoML`.

```python
model = ac.AutoML("Survived").fit(train)

report = model.report(test)              # == ac.evaluate_model(model, test)
print(report)                            # accuracy, F1, ROC-AUC, PR-AUC, MCC, log loss, Brier, ECE ...
report.confusion_matrix                  # also: per_class, calibration, roc_curve, pr_curve
report.confidence_intervals              # 95% bootstrap intervals for the headline metrics
report.best_threshold("f1")              # threshold table for binary models -> best cut-off
ac.evaluate_model(model, test, threshold=0.3)   # metrics at a different decision threshold
```

Regression reports give RMSE, MAE, median/max error, R², bias, MAPE/sMAPE, the residual
distribution and the error for each range of the actual value.

**SHAP explanations.** Shapley values are computed by the library itself, with no `shap` dependency. They work for any
model, including AutoML's ensembles and preprocessing. With up to 8 features every
coalition is evaluated (exact SHAP); otherwise antithetic permutation sampling is used. For every
row `base_value + sum(values)` equals the model's output exactly.

```python
exp = model.explain(test, rows=100)      # == ac.explain(model, test)
print(exp)                               # global importance (mean |SHAP|) + direction of effect
exp.global_importance()                  # feature, mean_abs_shap, share, direction
exp.explain_row(0)                       # "P(Survived = 1) = 0.27 (average 0.42): Pclass = 3 (-0.174), ..."
exp.for_row(0)                           # contributions for one row, largest first
exp.to_shap()                            # a shap.Explanation for shap's plots: pip install "AutomatedCleaning[explain]"
ac.partial_dependence(model, test, "Fare")   # average prediction as one feature varies
```

**Error analysis** finds the worst predictions and the segments of the data where the model
fails. A segment is a category, a quantile bin, missing values, or a pair of these. Each
segment is tested against the rest of the data, with a Benjamini–Hochberg correction so random
noise isn't flagged.

```python
errors = model.error_analysis(test)
errors.problem_slices        # e.g. "region = east": error rate 31% (2.4x), q = 0.001
errors.worst                 # most confident mistakes / largest errors, with their inputs
errors.confusions            # most common actual -> predicted mistakes
```

**Fairness & bias checks.** These cover each sensitive attribute, intersections such as sex × age
band, and *proxy* features that encode a sensitive attribute.

```python
fair = model.fairness(test, ["Sex", "Age"])      # numeric attributes are cut into quantile bands
print(fair)            # selection rate, TPR, FPR, precision and accuracy per group, plus flags
fair.passed            # False if any limit is exceeded
fair["Sex"].groups     # the per-group table
fair.proxies           # Cramér's V between every feature and each sensitive attribute
```

The checks cover demographic parity, disparate impact (four-fifths rule), equal opportunity,
equalized odds, predictive parity and accuracy gaps. For regression, they compare error and bias
across groups. The default limits are in `ac.evaluation.DEFAULT_LIMITS`; override them with
`limits={...}`.

**Model cards.** One call generates a full card in the standard layout (Mitchell et al.):
- model details and training data;
- the model-selection leaderboard;
- held-out metrics with confidence intervals and calibration;
- SHAP importance, error analysis and fairness;
- the quality score of the evaluation data;
- auto-written limitations and ethical considerations;
- versions.

```python
card = model.model_card(test, sensitive=["Sex"], name="Titanic survival",
                        intended_use="Teaching example; not for real decisions.", owners="Data team")
card.save("model_card.html")     # self-contained page (light/dark); also .md and .json
card.limitations                 # e.g. "Performs significantly worse on some segments: ..."
```
