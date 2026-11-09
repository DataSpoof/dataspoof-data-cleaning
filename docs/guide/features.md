# Feature engineering


Feature steps work like the cleaning steps: they learn in `fit`, replay in `transform`, and save
as JSON. Give the pipeline a `target` so steps that need the label can use it:

```python
pipe = ac.Pipeline(
    [*ac.default_cleaning_pipeline().steps, *ac.default_feature_pipeline("churned").steps],
    target="churned",
)
train_features = pipe.fit_transform(train_df)       # model-ready numeric features + the label
new_features = pipe.transform(new_df)               # the label isn't needed to predict
```

| Step | What it adds |
|---|---|
| `DateTimeFeatures` | year, month, day, weekday, quarter, weekend/month-start/month-end flags, hour, sine/cosine encodings, days elapsed |
| `TargetEncode` | each category's average target value. Training rows are encoded **out-of-fold**, so a row never sees its own label |
| `Interactions` | products and ratios of the most informative numeric columns |
| `GroupAggregations` | per-group statistics such as `fare_mean_by_class` (never of the target) |
| `LagFeatures` | time-series lags and rolling means, per entity, in time order |
| `TextEmbeddings` | TF-IDF + SVD vectors (built in), or `method="sentence_transformers"` |
| `SelectFeatures` | keep features by random-forest importance, mutual information or variance |
| `DropLeakage` | detect and drop columns that leak the target |

The default pipeline runs these steps in order:

1. date/time features
2. drop leaky columns
3. text embeddings
4. target encoding for columns with 16+ categories
5. one-hot encoding for the rest
6. feature selection

`time_col="..."` adds lag features and turns off target encoding.

**Target-aware pipelines:** with `target=`, ordinary steps never see the label. That means
imputation can't invent labels, outlier clipping can't change them, and encoders can't encode
them. Row-removing steps drop the same rows from the label, and the label can be absent at
prediction time.

**Leakage detection:** it scores every column on its own against the target. A column that
predicts almost perfectly is usually recorded *after* the outcome (`refund_issued` when
predicting churn), and such columns are flagged:

```python
report = ac.detect_leakage(df, "churned")
print(report)            # [high] refund_issued: on its own it predicts 'churned' with cv roc_auc = 1.000 ...
```
