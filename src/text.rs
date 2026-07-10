//! Text preprocessing ported from the Python `preprocess_text` function.
//!
//! Mirrors the original cleaning steps: lowercase, contraction expansion,
//! URL / mention / special-character / emoji stripping, whitespace tokenization
//! and (optional) stopword removal. Lemmatization is intentionally left to the
//! Python frontend so that NLTK's WordNet lemmatizer output is preserved exactly.

use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::{HashMap, HashSet};

/// Contraction / slang expansion table (identical to the Python `contractions`).
static CONTRACTIONS: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    [
        ("ain't", "am not"), ("aren't", "are not"), ("can't", "cannot"),
        ("can't've", "cannot have"), ("'cause", "because"), ("could've", "could have"),
        ("couldn't", "could not"), ("couldn't've", "could not have"), ("didn't", "did not"),
        ("doesn't", "does not"), ("don't", "do not"), ("hadn't", "had not"),
        ("hadn't've", "had not have"), ("hasn't", "has not"), ("haven't", "have not"),
        ("he'd", "he would"), ("he'd've", "he would have"), ("he'll", "he will"),
        ("he's", "he is"), ("how'd", "how did"), ("how'll", "how will"),
        ("how's", "how is"), ("i'd", "i would"), ("i'll", "i will"), ("i'm", "i am"),
        ("i've", "i have"), ("isn't", "is not"), ("it'd", "it would"), ("it'll", "it will"),
        ("it's", "it is"), ("let's", "let us"), ("ma'am", "madam"), ("mayn't", "may not"),
        ("might've", "might have"), ("mightn't", "might not"), ("must've", "must have"),
        ("mustn't", "must not"), ("needn't", "need not"), ("oughtn't", "ought not"),
        ("shan't", "shall not"), ("sha'n't", "shall not"), ("she'd", "she would"),
        ("she'll", "she will"), ("she's", "she is"), ("should've", "should have"),
        ("shouldn't", "should not"), ("that'd", "that would"), ("that's", "that is"),
        ("there'd", "there had"), ("there's", "there is"), ("they'd", "they would"),
        ("they'll", "they will"), ("they're", "they are"), ("they've", "they have"),
        ("wasn't", "was not"), ("we'd", "we would"), ("we'll", "we will"),
        ("we're", "we are"), ("we've", "we have"), ("weren't", "were not"),
        ("what'll", "what will"), ("what're", "what are"), ("what's", "what is"),
        ("what've", "what have"), ("where'd", "where did"), ("where's", "where is"),
        ("who'll", "who will"), ("who's", "who is"), ("won't", "will not"),
        ("wouldn't", "would not"), ("you'd", "you would"), ("you'll", "you will"),
        ("you're", "you are"), ("wfh", "work from home"), ("wfo", "work from office"),
        ("idk", "i do not know"), ("brb", "be right back"), ("btw", "by the way"),
        ("tbh", "to be honest"), ("omw", "on my way"), ("lmk", "let me know"),
        ("fyi", "for your information"), ("imo", "in my opinion"),
        ("smh", "shaking my head"), ("nvm", "never mind"), ("ikr", "i know right"),
        ("fr", "for real"), ("rn", "right now"), ("gg", "good game"),
        ("dm", "direct message"), ("afaik", "as far as i know"),
        ("bff", "best friends forever"), ("ftw", "for the win"), ("hmu", "hit me up"),
        ("ggwp", "good game well played"),
    ]
    .into_iter()
    .collect()
});

static RE_URL: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?m)https?://.*[\r\n]*").unwrap());
static RE_MENTION: Lazy<Regex> = Lazy::new(|| Regex::new(r"@[A-Za-z0-9]+").unwrap());
static RE_SPECIAL: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"[_"\-;%()|+&=*%,!?:#$@\[\]/]"#).unwrap());
static RE_AHREF: Lazy<Regex> = Lazy::new(|| Regex::new(r"<a href").unwrap());
static RE_AMP: Lazy<Regex> = Lazy::new(|| Regex::new(r"&amp;").unwrap());
static RE_BR: Lazy<Regex> = Lazy::new(|| Regex::new(r"<br />").unwrap());
static RE_QUOTES: Lazy<Regex> = Lazy::new(|| Regex::new(r#"'""#).unwrap());
static RE_EMOJI: Lazy<Regex> = Lazy::new(|| {
    Regex::new(concat!(
        r"[\x{1F600}-\x{1F64F}\x{1F300}-\x{1F5FF}\x{1F680}-\x{1F6FF}",
        r"\x{1F700}-\x{1F77F}\x{1F780}-\x{1F7FF}\x{1F800}-\x{1F8FF}",
        r"\x{1F900}-\x{1F9FF}\x{1FA00}-\x{1FA6F}\x{1FA70}-\x{1FAFF}",
        r"\x{2702}-\x{27B0}\x{24C2}-\x{1F251}]+"
    ))
    .unwrap()
});

/// Clean a single text value. Returns space-joined tokens (pre-lemmatization).
pub fn preprocess_one(text: &str, stopwords: &HashSet<String>, remove_stopwords: bool) -> String {
    // 1. lowercase
    let lowered = text.to_lowercase();

    // 2. contraction expansion, token by token on whitespace
    let expanded: Vec<&str> = lowered
        .split_whitespace()
        .map(|w| *CONTRACTIONS.get(w).unwrap_or(&w))
        .collect();
    let joined = expanded.join(" ");

    // 3. sequential regex substitutions (same order as the Python source)
    let s = RE_URL.replace_all(&joined, "");
    let s = RE_MENTION.replace_all(&s, "");
    let s = RE_SPECIAL.replace_all(&s, " ");
    let s = RE_AHREF.replace_all(&s, " ");
    let s = RE_AMP.replace_all(&s, "");
    let s = RE_BR.replace_all(&s, " ");
    let s = RE_QUOTES.replace_all(&s, " ");
    let s = RE_EMOJI.replace_all(&s, "");

    // 4. whitespace tokenization + optional stopword removal
    let tokens: Vec<&str> = s
        .split_whitespace()
        .filter(|w| !remove_stopwords || !stopwords.contains(*w))
        .collect();

    tokens.join(" ")
}
