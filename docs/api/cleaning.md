# Data cleaning: API reference

Generated from the source docstrings. For a task-oriented introduction, read the [Data cleaning guide](../guide/cleaning.md).

**On this page:**

- [`mlpilot.cleaners`](#mlpilotcleaners): `DropColumns`, `DropCollinear`, `ConvertNumericText`, `ParseDates`, `Impute`, `ClipOutliers`, `RemoveOutliers`, `LogTransformSkewed`, `Scale`, `GroupRareCategories`, `EncodeCategorical`, `DropDuplicates`, `default_cleaning_pipeline`
- [`mlpilot.cleaning`](#mlpilotcleaning): `print_section_header`, `print_header`, `load_data`, `preprocess_text`, `detect_column_types_and_process_text`, `check_data_types`, `replace_symbols_and_convert_to_float`, `replace_symbols`, `fix_incorrect_data_types`, `fix_spelling_errors_in_columns`, `fix_spelling_errors_in_categorical`, `correct_categorical_values`, `manual_correction`, `handle_negative_values` …
- [`mlpilot.steps`](#mlpilotsteps): `standardize_column_names`, `drop_empty_rows_and_columns`, `drop_high_missing_columns`, `remove_duplicate_columns`, `drop_id_like_columns`, `replace_disguised_missing`, `add_missing_indicators`, `impute_missing`, `normalize_whitespace`, `normalize_unicode`, `fix_mojibake`, `standardize_casing`, `standardize_categories`, `parse_dates` …

---

## `mlpilot.cleaners`

<sub>Source: [`mlpilot/cleaners.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/cleaners.py)</sub>

Stateful cleaning steps built on the pipeline core.

Each step learns its rules in `fit` (stored as plain JSON in `state_`) and applies only
those rules in `transform`. Cleaning a test set or a production batch therefore never
learns from that batch, and a saved pipeline reproduces training-time cleaning exactly.

Quick start:
```
pipe = ac.default_cleaning_pipeline()
train_clean = pipe.fit_transform(train_df)
pipe.save("cleaning.json")
test_clean = ac.Pipeline.load("cleaning.json").transform(test_df)
```

### class `DropColumns`

```python
class DropColumns(Step):
```

Learn which columns to drop (mostly missing, constant, ID-like, duplicated) and drop them.

**Constructor**

```python
DropColumns(*, max_missing=0.5, drop_constant=True, drop_id_like=True, id_uniqueness=0.95, drop_duplicate_columns=True, columns=None, keep=None, fit_only=None)
```

### class `DropCollinear`

```python
class DropCollinear(Step):
```

Learn numeric columns that are near-copies of another (|correlation| > threshold) and drop them.

**Constructor**

```python
DropCollinear(*, threshold=0.95, columns=None, fit_only=None)
```

### class `ConvertNumericText`

```python
class ConvertNumericText(Step):
```

Learn text columns that hold numbers ("1,200", "$50", "(30)") and convert them, keeping signs.

**Constructor**

```python
ConvertNumericText(*, min_ratio=0.9, columns=None, fit_only=None)
```

### class `ParseDates`

```python
class ParseDates(Step):
```

Learn text columns that hold dates and parse them to native date/datetime types.

**Constructor**

```python
ParseDates(*, min_success=0.8, columns=None, fit_only=None)
```

### class `Impute`

```python
class Impute(Step):
```

Learn fill values on training data (median/mean/mode/constant) and reuse them on new data.

**Constructor**

```python
Impute(*, numeric='median', categorical='mode', fill_value_numeric=0, fill_value_categorical='missing', add_indicators=False, indicator_min_missing=0.0, columns=None, fit_only=None)
```

### class `ClipOutliers`

```python
class ClipOutliers(Step):
```

Learn outlier fences (IQR or quantiles) and cap values to them. Keeps every row.

**Constructor**

```python
ClipOutliers(*, method='iqr', k=1.5, lower_quantile=0.01, upper_quantile=0.99, columns=None, fit_only=None)
```

### class `RemoveOutliers`

```python
class RemoveOutliers(Step):
```

Learn IQR fences and drop rows outside them. Runs only while fitting unless fit_only=False.

**Constructor**

```python
RemoveOutliers(*, k=1.5, columns=None, fit_only=None)
```

#### `RemoveOutliers.row_mask()`

```python
row_mask(df)
```

### class `LogTransformSkewed`

```python
class LogTransformSkewed(Step):
```

Learn non-negative, right-skewed numeric columns and apply log1p to them.

**Constructor**

```python
LogTransformSkewed(*, threshold=1.0, columns=None, fit_only=None)
```

### class `Scale`

```python
class Scale(Step):
```

Learn scaling parameters (standard, min-max or robust) on training data and apply them.

**Constructor**

```python
Scale(*, method='standard', columns=None, skip_binary=True, fit_only=None)
```

### class `GroupRareCategories`

```python
class GroupRareCategories(Step):
```

Learn each column's common categories; rare and unseen values become `other_label`.

**Constructor**

```python
GroupRareCategories(*, min_frequency=0.01, other_label='Other', max_unique=100, columns=None, fit_only=None)
```

### class `EncodeCategorical`

```python
class EncodeCategorical(Step):
```

Learn category codes on training data: one-hot, ordinal or frequency encoding.

**Constructor**

```python
EncodeCategorical(*, method='onehot', columns=None, max_categories=20, max_unique=100, drop_original=True, fit_only=None)
```

### class `DropDuplicates`

```python
class DropDuplicates(Step):
```

Drop duplicate rows. Runs only while fitting unless fit_only=False.

**Constructor**

```python
DropDuplicates(*, subset=None, keep='first', fit_only=None)
```

### `default_cleaning_pipeline()`

```python
def default_cleaning_pipeline(name: str='automated-cleaning', *, on_schema_mismatch: str='warn', verbose: bool=False, audit_log=None, **overrides) -> Pipeline
```

The recommended cleaning pipeline: no prompts, fit once, replay anywhere.

Steps: fix missing-value placeholders → normalise text → repair numbers/dates stored as
text → drop duplicate rows (training only) → drop useless columns → impute (+ "was
missing" flags for columns ≥5% missing) → group rare categories → cap extreme outliers
(3×IQR) → drop near-duplicate numeric columns.

Override a step by name: `clip_outliers=False` removes it,
`impute={"numeric": "mean"}` changes its parameters.

---

## `mlpilot.cleaning`

<sub>Source: [`mlpilot/cleaning.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/cleaning.py)</sub>

### `print_section_header()`

```python
def print_section_header(title)
```

Prints a formatted section header with dashes.

### `print_header()`

```python
def print_header(title)
```

Prints a formatted section header with ASCII art font centered in the terminal.

### `load_data()`

```python
def load_data(file_path)
```

Load data from different formats using polars.

### `preprocess_text()`

```python
def preprocess_text(text, remove_stopwords=True)
```

Clean a single text value.

The heavy work (lowercasing, contraction expansion, URL/mention/special-char/
emoji stripping, tokenization and stopword removal) runs in the Rust backend.
WordNet lemmatization stays in Python so its output matches the original.

### `detect_column_types_and_process_text()`

```python
def detect_column_types_and_process_text(df)
```

Detect whether string columns in a Polars DataFrame are likely categorical, text, or JSON.

This function categorizes string columns as:
- 'categorical': Columns with relatively few unique values compared to total rows
- 'text': Columns with many unique values and longer string lengths
- 'json': Columns containing valid JSON objects/arrays

Parameters:
-----------
df : pl.DataFrame
    The input Polars DataFrame to analyze

Returns:
--------
tuple:
    (DataFrame with analysis results, set of categorical columns, set of text columns, set of json columns)

### `check_data_types()`

```python
def check_data_types(df)
```

Check the data types of columns.

### `replace_symbols_and_convert_to_float()`

```python
def replace_symbols_and_convert_to_float(df)
```

Replace symbols and convert to float using Polars.

### `replace_symbols()`

```python
def replace_symbols(df)
```

Strip currency/separator symbols ($ ₹ , - •) from string columns.

Symbol removal runs in the Rust backend (parallel, per value).

### `fix_incorrect_data_types()`

```python
def fix_incorrect_data_types(df)
```

### `fix_spelling_errors_in_columns()`

```python
def fix_spelling_errors_in_columns(df)
```

Fix spelling errors in column names by interacting with the user.

### `fix_spelling_errors_in_categorical()`

```python
def fix_spelling_errors_in_categorical(df)
```

Fix spelling errors in categorical columns using Claude AI or manual input.

### `correct_categorical_values()`

```python
def correct_categorical_values(model, column_name: str, values: list) -> list
```

Corrects categorical column values using Claude AI.

### `manual_correction()`

```python
def manual_correction(column_name: str, values: list) -> list
```

Allows user to manually correct categorical values.

### `handle_negative_values()`

```python
def handle_negative_values(df)
```

Handle negative values by printing column names with negatives and replacing them with absolute values.

### `handle_missing_values()`

```python
def handle_missing_values(df)
```

Handle missing values with visualization.

### `handle_duplicates()`

```python
def handle_duplicates(df)
```

Handle duplicate records.

### `check_outliers()`

```python
def check_outliers(df)
```

Check for outliers using IQR method.

### `remove_outliers()`

```python
def remove_outliers(df)
```

Remove outliers using the IQR method for numerical columns (Rust backend).

### `check_problem_type()`

```python
def check_problem_type(df, target_col)
```

Determine whether the problem is classification or regression.

### `check_and_handle_imbalance()`

```python
def check_and_handle_imbalance(df, target_col)
```

Check for class imbalance and handle it using user-selected undersampling or oversampling.

Parameters:
- df (pl.DataFrame): The input Polars dataframe
- target_col (str): The name of the target column

Returns:
- pl.DataFrame: A balanced dataframe or the original if skipped

### `check_skewness()`

```python
def check_skewness(df)
```

Check skewness in numerical columns.

### `fix_skewness()`

```python
def fix_skewness(df)
```

Fix skewness using log transformation (skewness computed in Rust).

### `check_multicollinearity()`

```python
def check_multicollinearity(df, threshold=0.7)
```

Check for multicollinearity and drop highly correlated features.

The correlation matrix and drop-list are computed in the Rust backend.

### `check_cardinality()`

```python
def check_cardinality(df: pl.DataFrame)
```

Check the cardinality (number of unique values) of categorical columns and remove columns with only one unique value.

Parameters:
    df (pl.DataFrame): The input DataFrame.

Returns:
    pl.DataFrame: The DataFrame after removing low-cardinality columns.
  dict: A dictionary with column names as keys and their cardinality as values.

### `save_cleaned_data()`

```python
def save_cleaned_data(df: pl.DataFrame, file_name='cleaned_data.csv', quantize=True)
```

Save cleaned DataFrame to a CSV file, optionally with compact (value-preserving) numeric types.

Parameters:
- df (pl.DataFrame): The input Polars dataframe.
- file_name (str): The name of the CSV file.
- quantize (bool): Whether to shrink numeric types without changing values (default: True).

Returns:
- None

### `save_dashboard()`

```python
def save_dashboard(html_content, filename='dashboard.html')
```

### `generate_dashboard()`

```python
def generate_dashboard(df: pl.DataFrame, background_image_path: str=None)
```

Render an interactive, enterprise-style EDA dashboard.

Writes `output/eda/dashboard.html` with a collapsible sidebar (Overview /
Univariate / Bivariate / Multivariate), a hamburger toggle, KPI cards, an
interactive data-quality panel, a data preview, and Plotly charts. The
Univariate tab has a column picker; the Bivariate tab has X / Y / Color
pickers that plot the chosen pair on demand. Charts lazy-render per section.

### `fix_json_columns()`

```python
def fix_json_columns(df: pl.DataFrame) -> pl.DataFrame
```

Detect and fix JSON-type columns in the Polars DataFrame.

### `detect_and_mask_pii_polars()`

```python
def detect_and_mask_pii_polars(df: pl.DataFrame, sample_size: int=10) -> pl.DataFrame
```

Detects and masks PII in a Polars DataFrame.

Args:
    df (pl.DataFrame): Input Polars DataFrame.
  sample_size (int): Number of rows to sample for PII detection.

Returns:
    pl.DataFrame: DataFrame with masked PII columns added.

### `clean_data()`

```python
def clean_data(df, background_image_path=None)
```

Main function to clean the data.

---

## `mlpilot.steps`

<sub>Source: [`mlpilot/steps.py`](https://github.com/DataSpoof/dataspoof-mlpilot/blob/main/mlpilot/steps.py)</sub>

Additional data-cleaning steps for MLPilot.

Every function takes a Polars DataFrame and returns a Polars DataFrame (a few
also return a report dict). Compute-heavy per-cell string work is delegated to
the Rust backend (`mlpilot._rustcore`); the rest uses Polars,
scikit-learn and a few optional libraries that are imported lazily so this
module always imports even if they are absent.

Grouped as: column/structure, missing values, string normalization, dtypes,
categorical, outliers, duplicates, scaling, validation, domain, and text.

### `standardize_column_names()`

```python
def standardize_column_names(df: pl.DataFrame) -> pl.DataFrame
```

Trim, lowercase, snake_case and de-duplicate column names.

### `drop_empty_rows_and_columns()`

```python
def drop_empty_rows_and_columns(df: pl.DataFrame) -> pl.DataFrame
```

Drop columns that are entirely null and rows where every value is null.

### `drop_high_missing_columns()`

```python
def drop_high_missing_columns(df: pl.DataFrame, threshold: float=0.5) -> pl.DataFrame
```

Drop columns whose missing fraction exceeds `threshold` (0-1).

### `remove_duplicate_columns()`

```python
def remove_duplicate_columns(df: pl.DataFrame) -> pl.DataFrame
```

Remove columns that hold identical values under different names.

### `drop_id_like_columns()`

```python
def drop_id_like_columns(df: pl.DataFrame, uniqueness: float=0.99) -> pl.DataFrame
```

Drop near-unique identifier columns (unique ratio >= `uniqueness`).

### `replace_disguised_missing()`

```python
def replace_disguised_missing(df: pl.DataFrame, extra_sentinels=None, numeric_sentinels=None) -> pl.DataFrame
```

Convert disguised-missing tokens (`"NA"`, `"?"`, `999` ...) to null.

String columns are handled in the Rust backend; `numeric_sentinels` (e.g.
`[999, -1]`) are nulled out in numeric columns when provided.

### `add_missing_indicators()`

```python
def add_missing_indicators(df: pl.DataFrame, suffix: str='_missing') -> pl.DataFrame
```

Add a boolean `<col><suffix>` flag for every column that has nulls.

### `impute_missing()`

```python
def impute_missing(df: pl.DataFrame, numeric_strategy: str='median', categorical_strategy: str='mode', constant_value=0, knn_neighbors: int=5, columns=None) -> pl.DataFrame
```

Impute missing values with a selectable strategy.

numeric_strategy: `mean|median|knn|mice|interpolate|ffill|bfill|constant`
categorical_strategy: `mode|ffill|bfill|constant`

### `normalize_whitespace()`

```python
def normalize_whitespace(df: pl.DataFrame, columns=None) -> pl.DataFrame
```

Trim and collapse internal whitespace in string columns (Rust).

### `normalize_unicode()`

```python
def normalize_unicode(df: pl.DataFrame, columns=None) -> pl.DataFrame
```

NFKC-normalize and strip control/non-printable characters (Rust).

### `fix_mojibake()`

```python
def fix_mojibake(df: pl.DataFrame, columns=None) -> pl.DataFrame
```

Repair UTF-8-as-Latin-1 mojibake (e.g. `Ã©` -> `é`) (Rust).

### `standardize_casing()`

```python
def standardize_casing(df: pl.DataFrame, mapping: dict) -> pl.DataFrame
```

Standardize casing per column. `mapping`: col -> `title|lower|upper`.

### `standardize_categories()`

```python
def standardize_categories(df: pl.DataFrame, columns=None) -> pl.DataFrame
```

Trim + lowercase categorical values so duplicates actually collapse (Rust).

### `parse_dates()`

```python
def parse_dates(df: pl.DataFrame, columns=None, min_success: float=0.7) -> pl.DataFrame
```

Detect and parse date/datetime string columns to native dtypes.

A column is converted only if at least `min_success` of its non-null
values parse successfully.

### `extract_date_parts()`

```python
def extract_date_parts(df: pl.DataFrame, columns=None) -> pl.DataFrame
```

Add year/month/day/weekday/quarter columns for each date/datetime column.

### `normalize_booleans()`

```python
def normalize_booleans(df: pl.DataFrame, columns=None) -> pl.DataFrame
```

Convert boolean-like string columns (yes/no, Y/N, 1/0 ...) to real booleans (Rust).

### `downcast_numeric()`

```python
def downcast_numeric(df: pl.DataFrame) -> pl.DataFrame
```

Shrink numeric columns to the smallest dtype that fits (int64->int32 ...).

Types are chosen per call, so two batches can come out with different types, and floats
can lose digits. For value-preserving types learned once and reused, use
`mlpilot.compression.optimize_dtypes` or the `OptimizeDtypes` pipeline step.

### `group_rare_categories()`

```python
def group_rare_categories(df: pl.DataFrame, threshold: float=0.01, other_label: str='Other', columns=None) -> pl.DataFrame
```

Replace categories rarer than `threshold` (fraction of rows) with `other_label`.

### `fuzzy_cluster_categories()`

```python
def fuzzy_cluster_categories(df: pl.DataFrame, columns=None, threshold: int=90) -> pl.DataFrame
```

Merge near-identical category spellings to a canonical (most frequent) value.

Uses rapidfuzz token-set ratio; values >= `threshold` similar are merged.

### `encode_categorical()`

```python
def encode_categorical(df: pl.DataFrame, method: str='onehot', columns=None, target: str=None, max_onehot: int=20) -> pl.DataFrame
```

Encode categorical columns.

method: `onehot|label|frequency|ordinal|target` (`target` requires `target`).

### `remove_outliers_zscore()`

```python
def remove_outliers_zscore(df: pl.DataFrame, threshold: float=3.0, columns=None) -> pl.DataFrame
```

Remove rows where any numeric column's z-score exceeds `threshold`.

### `remove_outliers_mad()`

```python
def remove_outliers_mad(df: pl.DataFrame, threshold: float=3.5, columns=None) -> pl.DataFrame
```

Remove rows using the modified z-score (median absolute deviation).

### `remove_outliers_isolation_forest()`

```python
def remove_outliers_isolation_forest(df: pl.DataFrame, contamination: float=0.05, columns=None) -> pl.DataFrame
```

Remove multivariate outliers with an Isolation Forest over numeric columns.

### `winsorize()`

```python
def winsorize(df: pl.DataFrame, lower: float=0.05, upper: float=0.05, columns=None) -> pl.DataFrame
```

Cap (rather than drop) numeric values at the given lower/upper quantiles.

### `remove_outliers_by_group()`

```python
def remove_outliers_by_group(df: pl.DataFrame, group_col: str, columns=None, k: float=1.5) -> pl.DataFrame
```

Remove IQR outliers computed *within* each group of `group_col`.

### `drop_duplicates_subset()`

```python
def drop_duplicates_subset(df: pl.DataFrame, subset=None, keep: str='first') -> pl.DataFrame
```

Drop duplicate rows on an optional `subset` of key columns.

### `fuzzy_deduplicate()`

```python
def fuzzy_deduplicate(df: pl.DataFrame, key_columns, threshold: int=90, keep: str='first') -> pl.DataFrame
```

Remove near-duplicate rows by fuzzy-matching a concatenated key.

Rows whose key is >= `threshold` similar to an already-kept row are dropped.
O(n²) — intended for moderate row counts / after blocking.

### `scale_numeric()`

```python
def scale_numeric(df: pl.DataFrame, method: str='standard', columns=None) -> pl.DataFrame
```

Scale numeric columns: `minmax|standard|robust`.

### `bin_numeric()`

```python
def bin_numeric(df: pl.DataFrame, columns, bins: int=5, strategy: str='quantile') -> pl.DataFrame
```

Discretize numeric columns into `bins` buckets: `quantile|uniform`.

Adds a `<col>_bin` categorical column.

### `apply_range_rules()`

```python
def apply_range_rules(df: pl.DataFrame, rules: dict, action: str='null') -> pl.DataFrame
```

Enforce numeric ranges. `rules`: col -> (min, max). `action`: `null|clip|drop`.

### `check_cross_field()`

```python
def check_cross_field(df: pl.DataFrame, checks: dict, drop_violations: bool=False) -> pl.DataFrame
```

Check cross-field consistency rules.

`checks`: name -> Polars boolean expression that is True for VALID rows,
e.g. `{"date_order": pl.col("start") <= pl.col("end")}`. Reports violation
counts; optionally drops violating rows.

### `validate_formats()`

```python
def validate_formats(df: pl.DataFrame, specs: dict, add_flag: bool=True) -> pl.DataFrame
```

Validate string columns against a format. `specs`: col -> `email|url|ip|phone|postal`.

Adds a boolean `<col>_valid` column (when `add_flag`) and reports invalid counts.

### `check_referential_integrity()`

```python
def check_referential_integrity(df: pl.DataFrame, column: str, valid_values, drop_invalid: bool=False) -> pl.DataFrame
```

Report (optionally drop) rows whose `column` value is not in `valid_values`.

### `normalize_phone_numbers()`

```python
def normalize_phone_numbers(df: pl.DataFrame, columns, region: str='US') -> pl.DataFrame
```

Normalize phone numbers to E.164 (e.g. `+14155552671`) using `phonenumbers`.

### `normalize_emails()`

```python
def normalize_emails(df: pl.DataFrame, columns, add_flag: bool=True) -> pl.DataFrame
```

Lowercase/trim emails and flag validity (`email_validator` if available).

### `canonicalize_urls()`

```python
def canonicalize_urls(df: pl.DataFrame, columns) -> pl.DataFrame
```

Canonicalize URLs: add scheme, lowercase host, strip fragments/trailing slash.

### `standardize_country_region()`

```python
def standardize_country_region(df: pl.DataFrame, columns, mapping: dict=None) -> pl.DataFrame
```

Standardize country names/codes to a canonical form using a mapping table.

### `convert_currency()`

```python
def convert_currency(df: pl.DataFrame, columns, rates: dict, base: str='USD') -> pl.DataFrame
```

Convert numeric currency columns to `base` using `rates` (col -> multiplier).

### `detect_language()`

```python
def detect_language(df: pl.DataFrame, column: str, out_col: str=None) -> pl.DataFrame
```

Add a detected-language column (ISO code) using `langdetect` (optional).

### `filter_by_language()`

```python
def filter_by_language(df: pl.DataFrame, column: str, keep: str='en') -> pl.DataFrame
```

Keep only rows whose `column` text is detected as language `keep`.

### `correct_spelling_text()`

```python
def correct_spelling_text(df: pl.DataFrame, columns) -> pl.DataFrame
```

Correct spelling in free-text columns using `pyspellchecker` (optional).

### `redact_profanity()`

```python
def redact_profanity(df: pl.DataFrame, columns, replacement: str='****') -> pl.DataFrame
```

Redact profanity in text columns using `better_profanity` (optional).

### `near_duplicate_text()`

```python
def near_duplicate_text(df: pl.DataFrame, column: str, threshold: int=90, drop: bool=False) -> pl.DataFrame
```

Flag (or drop) rows whose `column` text is a near-duplicate of an earlier row.

Adds a boolean `<column>_near_dup` flag. O(n²) — use on moderate sizes.
