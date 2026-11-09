# Anomaly detection


```python
det = ac.AutoAnomaly().fit(transactions)
det.score(new)            # anomaly_score (0-1) and is_anomaly
det.explain(new)          # "amount = 9,800 is unusually high (8.9 robust SDs from typical 120)"
```

It combines three detectors, each catching a different kind of anomaly:
- **Isolation Forest:** rare combinations of values.
- **Local Outlier Factor:** points far from their neighbours.
- **Robust z-score:** extreme single values.

The detectors are combined on robustly standardised scores. The share of rows flagged is
chosen automatically, capped at 5%.
