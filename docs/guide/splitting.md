# Data splitting


```python
s = ac.split_data(df, target="churned", test_size=0.2, val_size=0.1)    # stratified for classification
s = ac.split_data(df, target="churned", group="customer_id")            # a customer is never on both sides
s = ac.split_data(df, target="sales", time="date", gap="7d")            # past -> train, future -> test
print(s.report())                       # sizes, class balance, time ranges, leakage checks
ac.check_split(my_train, my_test, group="patient_id", time="visit")     # audit a split made elsewhere
for train, test in ac.cv_folds(df, target="churned", n_splits=5, group="customer_id"): ...
```

**How the strategy is chosen:**
- A `time=` column gives a chronological split. `gap` drops the rows just before each boundary,
  so window features can't leak forward.
- A `group=` column keeps every row of a group in one split. With a classification target, the
  groups are also balanced by class.
- Otherwise the split is random, and stratified by the target for classification.

**Leakage checks:** every split is checked afterwards for
- groups appearing on both sides;
- test rows older than the newest training row;
- identical rows in train and test;
- test classes never seen in training.
