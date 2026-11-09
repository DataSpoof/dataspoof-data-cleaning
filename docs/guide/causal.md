# Causal inference & uplift


```python
est = ac.estimate_effect(df, treatment="got_discount", outcome="revenue")
print(est)     # effect +5.1 (95% CI +4.1 to +6.1), naive +13.5 - confounding changes the answer ...
up = ac.UpliftModel("got_offer", "converted").fit(campaign)
up.predict_uplift(customers)                     # who actually responds to the offer
ac.qini(up.predict_uplift(test), test["got_offer"], test["converted"])
```

**What `estimate_effect` does:** it estimates what a treatment, such as a discount, actually
*causes*, not just what it's associated with:
- **Method:** a doubly-robust estimate (AIPW) with cross-fitting.
- **Comparison:** it reports the naive difference next to the corrected estimate.
- **Diagnostics:** it checks overlap (whether treated and untreated customers are comparable)
  and covariate balance, and warns when the estimate shouldn't be trusted.

It states the assumption it rests on: every common cause of treatment and outcome is among the
confounders.
