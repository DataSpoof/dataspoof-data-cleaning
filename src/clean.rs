//! Per-cell string cleaning ported to Rust: whitespace normalization, Unicode
//! NFKC + control-char removal, boolean normalization, disguised-missing
//! detection and mojibake repair. All are applied in batch (parallel) from the
//! Python frontend.

use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashSet;
use unicode_normalization::UnicodeNormalization;

static RE_WS: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+").unwrap());

/// Trim ends and collapse internal runs of whitespace to a single space.
pub fn normalize_whitespace(s: &str) -> String {
    RE_WS.replace_all(s.trim(), " ").into_owned()
}

/// Apply NFKC normalization and drop control / non-printable characters.
pub fn normalize_unicode(s: &str) -> String {
    s.nfkc().filter(|c| !c.is_control()).collect()
}

/// Map a value to canonical "true"/"false" if it is a recognizable boolean,
/// otherwise `None` (so a boolean column's junk becomes null).
pub fn normalize_boolean(s: &str) -> Option<String> {
    match s.trim().to_lowercase().as_str() {
        "true" | "t" | "yes" | "y" | "1" | "on" => Some("true".to_string()),
        "false" | "f" | "no" | "n" | "0" | "off" => Some("false".to_string()),
        _ => None,
    }
}

/// Repair classic UTF-8-as-Latin-1 mojibake (e.g. "Ã©" -> "é") using the
/// round-trip heuristic: if every char fits in a byte, re-decode as UTF-8.
pub fn fix_mojibake(s: &str) -> String {
    // Only attempt when there are high-Latin-1 chars typical of mojibake.
    if !s.chars().any(|c| ('\u{80}'..='\u{FF}').contains(&c)) {
        return s.to_string();
    }
    let bytes: Option<Vec<u8>> = s
        .chars()
        .map(|c| {
            let u = c as u32;
            if u <= 0xFF {
                Some(u as u8)
            } else {
                None
            }
        })
        .collect();
    if let Some(b) = bytes {
        if let Ok(fixed) = String::from_utf8(b) {
            return fixed;
        }
    }
    s.to_string()
}

/// Return `true` if the (trimmed, lowercased) value is in the disguised-missing
/// sentinel set.
pub fn is_disguised_missing(s: &str, sentinels: &HashSet<String>) -> bool {
    sentinels.contains(s.trim().to_lowercase().as_str())
}
