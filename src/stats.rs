//! Numeric statistics ported from the Python cleaning routines: skewness,
//! IQR-based outlier bounds/masking, Pearson correlation and the
//! multicollinearity drop-list. All functions ignore NaN (used to represent
//! Polars nulls passed from Python).

/// Fisher-Pearson skewness with `bias=True`, matching Polars' `Series.skew()`
/// default: g1 = m3 / m2^(3/2).
pub fn skewness(values: &[f64]) -> f64 {
    let xs: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    let n = xs.len();
    if n == 0 {
        return f64::NAN;
    }
    let mean = xs.iter().sum::<f64>() / n as f64;
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    for &x in &xs {
        let d = x - mean;
        m2 += d * d;
        m3 += d * d * d;
    }
    m2 /= n as f64;
    m3 /= n as f64;
    if m2 == 0.0 {
        return f64::NAN;
    }
    m3 / m2.powf(1.5)
}

/// Quantile using "nearest" interpolation, matching Polars' default. Values
/// are sorted ascending; NaN is dropped first.
fn quantile_nearest(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let pos = q * (n as f64 - 1.0);
    let idx = pos.round() as usize;
    sorted[idx.min(n - 1)]
}

/// Return the (lower_bound, upper_bound) IQR fences for a column.
pub fn iqr_bounds(values: &[f64]) -> (f64, f64) {
    let mut xs: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q1 = quantile_nearest(&xs, 0.25);
    let q3 = quantile_nearest(&xs, 0.75);
    let iqr = q3 - q1;
    (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
}

/// Boolean keep-mask: `true` where the value is within the IQR fences.
/// NaN (null) rows are dropped (`false`), matching Polars comparison semantics.
pub fn iqr_keep_mask(values: &[f64]) -> Vec<bool> {
    let (lower, upper) = iqr_bounds(values);
    values
        .iter()
        .map(|&v| !v.is_nan() && v >= lower && v <= upper)
        .collect()
}

/// Pearson correlation between two columns, pairwise-ignoring NaN.
fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let pairs: Vec<(f64, f64)> = a
        .iter()
        .zip(b.iter())
        .filter(|(x, y)| !x.is_nan() && !y.is_nan())
        .map(|(x, y)| (*x, *y))
        .collect();
    let n = pairs.len();
    if n == 0 {
        return f64::NAN;
    }
    let mean_a = pairs.iter().map(|p| p.0).sum::<f64>() / n as f64;
    let mean_b = pairs.iter().map(|p| p.1).sum::<f64>() / n as f64;
    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for (x, y) in &pairs {
        let da = x - mean_a;
        let db = y - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    let denom = (var_a * var_b).sqrt();
    if denom == 0.0 {
        return f64::NAN;
    }
    cov / denom
}

/// Full Pearson correlation matrix for the given columns.
pub fn correlation_matrix(columns: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let k = columns.len();
    let mut m = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in 0..k {
            m[i][j] = if i == j { 1.0 } else { pearson(&columns[i], &columns[j]) };
        }
    }
    m
}

/// Replicates the Python multicollinearity check: walking the lower triangle,
/// mark column `i` for dropping when |corr(i, j)| > threshold for some j < i.
pub fn collinear_to_drop(columns: &[Vec<f64>], names: &[String], threshold: f64) -> Vec<String> {
    let corr = correlation_matrix(columns);
    let k = columns.len();
    let mut to_drop: Vec<String> = Vec::new();
    for i in 0..k {
        for j in 0..i {
            if corr[i][j].abs() > threshold {
                if !to_drop.contains(&names[i]) {
                    to_drop.push(names[i].clone());
                }
                break;
            }
        }
    }
    to_drop
}
