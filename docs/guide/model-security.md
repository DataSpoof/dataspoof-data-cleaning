# Model security


```python
rep = ac.robustness_report(model, test_df)      # any AutoML or scikit-learn model (target=...)
print(rep.summary())                            # noise stability, adversarial distance, sensitive features
rep.passed(max_noise_flip=0.05, min_median_distance=0.25)

pois = ac.poisoning_report(train_df, target="fraud")
print(pois.summary())                           # label errors, conflicting duplicates, triggers, outliers
clean = pois.clean(train_df)
```

**Robustness** uses black-box tests, so the model's own preprocessing is part of what is attacked:
- **Noise stability** at 1%, 5% and 10% of each feature's standard deviation.
- **Minimal adversarial change:** a greedy evasion attack measures, in standard deviations, how
  small a change flips each prediction, and which feature it exploits.
- **Per-feature sensitivity:** each feature is shifted, swapped or set to missing.
- **Extreme inputs** must not crash the model or produce NaN.

**Poisoning detection:**
- **Label errors:** confident learning on out-of-fold probabilities finds mislabelled rows.
- **Conflicts:** identical rows that carry different labels.
- **Backdoor triggers:** a rare value whose rows almost all share one label, tested with a
  Holm-corrected binomial test.
- **Outliers**, found with Isolation Forest.

**In testing:**
- At least 70% of deliberately flipped labels were found, with at least half of the flags correct.
- Both planted triggers were found: a categorical code and a repeated numeric value.
- Clean data produced no false triggers.
