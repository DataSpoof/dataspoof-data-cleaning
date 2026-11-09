# Feature store


```python
fs = ac.FeatureStore("feature_repo")
fs.apply(ac.Entity("customer", ["customer_id"]),
         ac.FeatureView("customer_stats", entities=["customer"], timestamp="as_of",
                        features=["orders_30d", "spend_30d"], ttl_days=30, owner="growth"))
fs.ingest("customer_stats", daily_snapshot)                           # history + online latest
train = fs.get_historical_features(labels, ["customer_stats:*"])      # point-in-time correct
fs.get_online_features(["customer_stats:spend_30d"], [{"customer_id": 42}])   # serving lookup
```

**No leakage in training data:** each training row gets feature values *as they were at that
row's timestamp*, never later. Values older than `ttl_days` are dropped.

**Storage:**
- The **offline** store is append-only Parquet, holding the full history for training.
- The **online** store is SQLite, holding the latest value per entity for serving. Late-arriving
  old rows never overwrite newer values. `materialize()` rebuilds it as of any date.

**Safety:** changing a feature's type, or removing a feature that already has data, is refused.

**Moving to Feast?** `integrations.feast.export_to_feast(store, "feast_repo")` writes and applies
an equivalent Feast repository. In testing, Feast's point-in-time joins and online lookups
matched these ones ([section 15](integrations.md)).
