//! Rust compute backend for the AutomatedCleaning library.
//!
//! Exposed to Python as the `automatedcleaning._rustcore` extension module.
//! Each function takes plain Python lists (converted automatically by PyO3) so
//! the Python frontend can hand columns straight from Polars/pandas.

mod stats;
mod text;

use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashSet;

/// Batch text preprocessing (parallel). Returns cleaned, space-joined tokens for
/// each input; lemmatization is applied afterwards on the Python side. `None`
/// / non-string inputs map to an empty string, matching the Python guard.
#[pyfunction]
#[pyo3(signature = (texts, stopwords, remove_stopwords=true))]
fn preprocess_texts(
    py: Python<'_>,
    texts: Vec<Option<String>>,
    stopwords: Vec<String>,
    remove_stopwords: bool,
) -> Vec<String> {
    let stop: HashSet<String> = stopwords.into_iter().collect();
    // Release the GIL so Rust threads can run truly in parallel.
    py.allow_threads(|| {
        texts
            .par_iter()
            .map(|t| match t {
                Some(s) => text::preprocess_one(s, &stop, remove_stopwords),
                None => String::new(),
            })
            .collect()
    })
}

/// Strip currency / separator symbols ($ ₹ , - •) from each value, in parallel.
/// Mirrors the Python `replace_symbols` regex `[\$,₹,-,•]`.
#[pyfunction]
fn strip_symbols(py: Python<'_>, texts: Vec<Option<String>>) -> Vec<Option<String>> {
    py.allow_threads(|| {
        texts
            .par_iter()
            .map(|t| {
                t.as_ref().map(|s| {
                    s.chars()
                        .filter(|c| !matches!(c, '$' | '₹' | ',' | '-' | '•'))
                        .collect::<String>()
                })
            })
            .collect()
    })
}

/// Fisher-Pearson skewness (matches Polars `Series.skew()`).
#[pyfunction]
fn skewness(values: Vec<f64>) -> f64 {
    stats::skewness(&values)
}

/// (lower, upper) IQR outlier fences for a numeric column.
#[pyfunction]
fn iqr_bounds(values: Vec<f64>) -> (f64, f64) {
    stats::iqr_bounds(&values)
}

/// Boolean keep-mask (within-fences) for IQR outlier removal.
#[pyfunction]
fn iqr_keep_mask(values: Vec<f64>) -> Vec<bool> {
    stats::iqr_keep_mask(&values)
}

/// Full Pearson correlation matrix for a list of numeric columns.
#[pyfunction]
fn correlation_matrix(columns: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    stats::correlation_matrix(&columns)
}

/// Column names to drop for multicollinearity above `threshold`.
#[pyfunction]
fn collinear_to_drop(columns: Vec<Vec<f64>>, names: Vec<String>, threshold: f64) -> Vec<String> {
    stats::collinear_to_drop(&columns, &names, threshold)
}

/// `true` if any value in the column is negative (NaN ignored).
#[pyfunction]
fn has_negative(values: Vec<f64>) -> bool {
    values.iter().any(|v| !v.is_nan() && *v < 0.0)
}

/// JSON-column heuristic: of the sampled non-null strings, count how many both
/// look like a JSON object/array and parse successfully; classify as JSON when
/// that is >= 50% of the sample. Mirrors the Python detection logic.
#[pyfunction]
fn detect_json_sample(samples: Vec<String>) -> bool {
    if samples.is_empty() {
        return false;
    }
    let mut json_count = 0usize;
    for value in &samples {
        let v = value.trim();
        let looks_json = (v.starts_with('{') && v.ends_with('}'))
            || (v.starts_with('[') && v.ends_with(']'));
        if looks_json && serde_json::from_str::<serde_json::Value>(v).is_ok() {
            json_count += 1;
        }
    }
    json_count > 0 && (json_count as f64 / samples.len() as f64) >= 0.5
}

#[pymodule]
fn _rustcore(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__doc__", "Rust compute backend for AutomatedCleaning.")?;
    m.add_function(wrap_pyfunction!(preprocess_texts, m)?)?;
    m.add_function(wrap_pyfunction!(strip_symbols, m)?)?;
    m.add_function(wrap_pyfunction!(skewness, m)?)?;
    m.add_function(wrap_pyfunction!(iqr_bounds, m)?)?;
    m.add_function(wrap_pyfunction!(iqr_keep_mask, m)?)?;
    m.add_function(wrap_pyfunction!(correlation_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(collinear_to_drop, m)?)?;
    m.add_function(wrap_pyfunction!(has_negative, m)?)?;
    m.add_function(wrap_pyfunction!(detect_json_sample, m)?)?;
    Ok(())
}
