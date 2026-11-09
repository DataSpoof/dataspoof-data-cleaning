# Profiling & data quality


### Automated EDA

```python
p = ac.profile(df)
print(p)                 # overview, every column's type and meaning, plain-English insights
p["Fare"].numeric        # min, max, mean, median, std, p01…p99, skew, outliers, …
p.to_frame()             # one row per column, ready to filter or export
p.to_json("profile.json")
```

Each column gets a **semantic type** describing what it holds, not just how it's stored:
`identifier`, `binary`, `categorical`, `numeric`, `discrete`, `text`, `email`, `phone`, `url`,
`datetime`, `numeric_text` (numbers stored as text), `datetime_text` (dates stored as text),
`constant` or `empty`. Insights are generated for findings such as:

- columns that are mostly empty
- identifier and constant columns
- strong correlations
- heavy skew
- columns dominated by a single value

### Quality score

```python
report = ac.check_quality(df)
print(report)                  # score, grade, dimension bars, weakest columns, top issues
report.score, report.grade     # e.g. 97.2, "A"
report.dimensions              # completeness / uniqueness / validity / consistency, each 0-100
report.weakest_columns()       # e.g. [("Cabin", 50.9), ("Age", 89.3)]
report.to_frame()              # every issue with severity, column and cells affected
```

The score combines four standard data-quality dimensions:

- **completeness:** real values, counting `"NA"`/`"?"` placeholders and `-999` codes as missing
- **uniqueness:** rows that aren't duplicates
- **validity:** values pass range, email, phone and calendar-date rules
- **consistency:** no stray whitespace, case variants or mixed date formats

The default weights are 0.35 / 0.15 / 0.30 / 0.20 and can be changed with `weights=`. Grades run
A ≥ 90, B ≥ 80, C ≥ 70, D ≥ 60, otherwise F. The overall score averages over every cell, so one
broken column barely moves it; `weakest_columns()` is there to catch that.

### Data contracts

Declare what valid data looks like and check every batch against it:

```yaml
# customers.contract.yaml
name: customers
extra_columns: warn            # allow | warn | error
no_duplicate_rows: true
unique: [customer_id]
columns:
  customer_id: {type: integer, required: true, unique: true}
  email:       {type: string, format: email, max_null_fraction: 0.01}
  age:         {type: number, min: 0, max: 120, warn_only: [min, max]}
  plan:        {type: string, allowed: [basic, pro, enterprise]}
```

```python
contract = ac.Contract.from_file("customers.contract.yaml")
result = contract.validate(batch_df)
print(result)                  # PASSED/FAILED with every broken rule and example values
result.raise_if_failed()       # raise ContractViolation in a job or CI

# No contract yet? Infer one from data you trust, review it, then commit it:
ac.Contract.infer(train_df).to_yaml("customers.contract.yaml")
```

Inferred contracts make structural rules errors (columns, types, required, unique) and
statistical rules warnings (ranges, allowed values, lengths, null fractions), because new
data legitimately drifts.

### Schema checks

```python
diff = ac.compare_schemas(train_df, new_df)
diff.removed, diff.added, diff.changed      # e.g. {"age": ["Float64", "String"]}
diff.compatible                             # False if a column vanished or changed type
```

### Enforce quality inside pipelines

```python
pipe = ac.Pipeline([
    ac.ValidateContract(),                       # infers a contract from training data on fit
    *ac.default_cleaning_pipeline().steps,
    ac.QualityGate(min_score=85, min_dimensions={"completeness": 90}),
])
pipe.fit(train_df)
pipe.save("pipeline.json")                       # the inferred contract is saved inside it
ac.Pipeline.load("pipeline.json").transform(new_batch)   # raises if the batch breaks it
```

Use `on_fail="warn"` on either step to log a warning instead of stopping. The EDA dashboard's
Overview tab is built on this module, so it shows the same score, weakest columns and issues.
