# Data versioning


```python
repo = ac.DataRepository("data_repo")
v1 = repo.commit(df, "customers", message="initial extract", source="s3://raw/customers.csv")
v2 = repo.commit(df2, "customers", message="Q2 refresh", tags=["gold"])
repo.log("customers")                                   # who, when, rows, hash, message
repo.diff("customers", 1, 2, key="customer_id")         # +added / -removed / ~changed rows, column changes
old = repo.checkout("customers", version=1)             # the exact data back
repo.checkout("customers", tag="gold", run=run)         # ...and record which version a run used
```

- **Storage:** content-addressed, zstd-compressed Parquet. Committing identical data again
  creates no new copy.
- **Integrity:** every checkout is verified against the stored file's SHA-256, so modified files
  are detected.
- **Tags** move between versions the way git tags do.
- **Lineage:** `commit(..., run=...)` and `checkout(..., run=...)` link dataset versions to
  experiment runs, in both directions.
- **Already using DVC?** `integrations.dvc` versions data in a git + DVC repository, with
  remotes, old revisions and lineage, and `export_repository` replays this history into it
  ([section 15](integrations.md)).
