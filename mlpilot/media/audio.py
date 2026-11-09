"""Audio data: audit recordings, preprocess them consistently, and extract features for AutoML.

WAV works out of the box (scipy); FLAC / OGG / MP3 … need ``pip install "mlpilot[audio]"``
(soundfile). Everything else — resampling, silence trimming, loudness normalisation, MFCCs and
spectral features — is implemented with numpy/scipy (no librosa needed).

* ``load_audio(path, sr=16000)``            mono float32 in [-1, 1], resampled
* ``scan_audio("calls/**/*.wav")``          duration, sample rate, channels, loudness, clipping,
                                              silence, DC offset; unreadable files kept with ``error``
* ``audio_quality_report(scan)``            corrupt, clipped, silent / mostly silent, too short,
                                              mixed sample rates, DC offset, duplicates
* ``AudioPreprocessor(sr=16000, duration=3)``  mono, resample, trim silence, normalise loudness,
                                              pre-emphasis, pad/trim to a fixed length; JSON config
* ``audio_features(paths)``                 MFCC / delta statistics, spectral centroid, bandwidth,
                                              roll-off, flatness, zero-crossing rate, RMS -> DataFrame
"""

from __future__ import annotations

import glob as _glob
import hashlib
import json
from math import gcd
from pathlib import Path

import numpy as np
import polars as pl

from .image import MediaIssue, MediaQualityReport

AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif")


def _paths(source) -> list:
    if isinstance(source, (str, Path)):
        s = str(source)
        if Path(s).is_dir():
            return sorted(str(p) for p in Path(s).rglob("*") if p.suffix.lower() in AUDIO_EXTS)
        return sorted(p for p in _glob.glob(s, recursive=True) if Path(p).suffix.lower() in AUDIO_EXTS)
    return [str(p) for p in source]


def read_audio(path):
    """``(samples float32 (n, channels) in [-1, 1], sample_rate)`` without any processing."""
    path = str(path)
    if path.lower().endswith(".wav"):
        from scipy.io import wavfile
        sr, data = wavfile.read(path)
        if data.dtype.kind == "i":
            data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
        elif data.dtype.kind == "u":                         # 8-bit WAV is unsigned
            data = (data.astype(np.float32) - 128.0) / 128.0
        data = data.astype(np.float32)
    else:
        try:
            import soundfile as sf
        except ImportError as exc:
            raise ImportError('Non-WAV audio needs soundfile: pip install "mlpilot[audio]"') from exc
        data, sr = sf.read(path, dtype="float32", always_2d=True)
    if data.ndim == 1:
        data = data[:, None]
    return data, int(sr)


def resample(signal: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return signal
    from scipy.signal import resample_poly
    g = gcd(int(sr), int(target))
    return resample_poly(signal, target // g, sr // g).astype(np.float32)


def load_audio(path, sr: int | None = 16000, mono: bool = True):
    """Read, mix to mono and resample. Returns ``(signal, sample_rate)``."""
    data, rate = read_audio(path)
    sig = data.mean(axis=1) if mono else data
    if sr is not None and rate != sr:
        sig = resample(sig, rate, sr)
        rate = sr
    return sig.astype(np.float32), rate


def _frames(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def _db(x):
    return 20 * np.log10(np.maximum(x, 1e-10))


def scan_audio(source, *, labels: dict | None = None, label_from_folder: bool = False,
               silence_db: float = -50.0) -> pl.DataFrame:
    rows = []
    for path in _paths(source):
        row = {"path": path, "label": None, "error": None, "sample_rate": None, "channels": None, "duration": None,
               "bytes": None, "sha256": None, "rms_dbfs": None, "peak": None, "clipped_ratio": None,
               "silence_ratio": None, "dc_offset": None}
        if labels is not None:
            row["label"] = labels.get(path)
        elif label_from_folder:
            row["label"] = Path(path).parent.name
        try:
            raw = Path(path).read_bytes()
            row["bytes"], row["sha256"] = len(raw), hashlib.sha256(raw).hexdigest()
            data, sr = read_audio(path)
            mono = data.mean(axis=1)
            row["sample_rate"], row["channels"], row["duration"] = sr, int(data.shape[1]), len(mono) / sr
            row["peak"] = float(np.abs(data).max()) if data.size else 0.0
            row["rms_dbfs"] = float(_db(np.sqrt(np.mean(mono ** 2)) if mono.size else 0.0))
            row["clipped_ratio"] = float(np.mean(np.abs(data) >= 0.999)) if data.size else 0.0
            row["dc_offset"] = float(mono.mean()) if mono.size else 0.0
            fr = _frames(mono, max(1, int(0.025 * sr)), max(1, int(0.010 * sr)))
            row["silence_ratio"] = float(np.mean(_db(np.sqrt((fr ** 2).mean(axis=1))) < silence_db))
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"[:200]
        rows.append(row)
    schema = {"path": pl.String, "label": pl.String, "error": pl.String, "sample_rate": pl.Int64, "channels": pl.Int64,
              "duration": pl.Float64, "bytes": pl.Int64, "sha256": pl.String, "rms_dbfs": pl.Float64,
              "peak": pl.Float64, "clipped_ratio": pl.Float64, "silence_ratio": pl.Float64, "dc_offset": pl.Float64}
    return pl.DataFrame(rows, schema=schema)


def audio_quality_report(scan: pl.DataFrame, *, min_duration: float = 0.5, clip_ratio: float = 0.001,
                         max_silence: float = 0.8, dc_limit: float = 0.05) -> MediaQualityReport:
    issues = []
    bad = scan.filter(pl.col("error").is_not_null())
    if bad.height:
        issues.append(MediaIssue("unreadable", "high", f"{bad.height} file(s) can't be decoded", bad["path"].to_list()))
    ok = scan.filter(pl.col("error").is_null())
    checks = [
        ("silent", "high", ok.filter(pl.col("peak") < 1e-4), "contain no signal"),
        ("clipping", "medium", ok.filter(pl.col("clipped_ratio") > clip_ratio), f"have > {clip_ratio:.1%} clipped samples (distorted)"),
        ("mostly_silence", "medium", ok.filter((pl.col("silence_ratio") > max_silence) & (pl.col("peak") >= 1e-4)),
         f"are more than {max_silence:.0%} silence (trim it)"),
        ("too_short", "medium", ok.filter(pl.col("duration") < min_duration), f"are shorter than {min_duration}s"),
        ("dc_offset", "low", ok.filter(pl.col("dc_offset").abs() > dc_limit), "have a DC offset (centre them)"),
    ]
    for name, sev, sel, text in checks:
        if sel.height:
            issues.append(MediaIssue(name, sev, f"{sel.height} file(s) {text}", sel["path"].to_list()))
    dup = ok.filter(pl.col("sha256").is_duplicated())
    if dup.height:
        issues.append(MediaIssue("exact_duplicates", "medium", f"{dup.height} files are identical copies", dup["path"].to_list()))
    rates = ok["sample_rate"].value_counts(sort=True)
    if rates.height > 1:
        issues.append(MediaIssue("mixed_sample_rates", "low",
                                 "sample rates " + ", ".join(f"{r} Hz x{n}" for r, n in rates.iter_rows())
                                 + " - resample to one rate (AudioPreprocessor does)", []))
    labels = ok.filter(pl.col("label").is_not_null())["label"].value_counts(sort=True)
    if labels.height >= 2 and labels["count"][0] >= 5 * labels["count"][-1]:
        issues.append(MediaIssue("label_imbalance", "medium",
                                 f"most common label has {labels['count'][0]} clips, rarest {labels['count'][-1]}", []))
    stats = {"readable": ok.height, "total_hours": float((ok["duration"].sum() or 0) / 3600),
             "median_duration": ok["duration"].median(), "sample_rates": dict(rates.iter_rows()) if rates.height else {}}
    return MediaQualityReport("audio", scan.height, issues, stats)


class AudioPreprocessor:
    """Mono -> resample -> trim leading/trailing silence -> loudness normalise -> pre-emphasis ->
    pad/trim to ``duration`` seconds. Stateless apart from its settings (saved as JSON)."""

    def __init__(self, sr: int = 16000, *, duration: float | None = None, trim_db: float | None = -40.0,
                 normalize: str | None = "rms", target_dbfs: float = -20.0, pre_emphasis: float | None = None):
        if normalize not in (None, "rms", "peak"):
            raise ValueError("normalize must be None, 'rms' or 'peak'")
        self.sr, self.duration, self.trim_db = sr, duration, trim_db
        self.normalize, self.target_dbfs, self.pre_emphasis = normalize, target_dbfs, pre_emphasis

    def process(self, signal: np.ndarray, sr: int) -> np.ndarray:
        x = signal.mean(axis=1) if signal.ndim == 2 else signal
        x = resample(x.astype(np.float32), sr, self.sr)
        x = x - x.mean() if x.size else x
        if self.trim_db is not None and x.size:
            fr = max(1, int(0.02 * self.sr))
            frames = _frames(x, fr, fr)
            loud = np.where(_db(np.sqrt((frames ** 2).mean(axis=1))) > self.trim_db)[0]
            x = x[loud[0] * fr: (loud[-1] + 1) * fr] if loud.size else x[:0]
        if self.normalize and x.size:
            target = 10 ** (self.target_dbfs / 20)
            level = np.sqrt(np.mean(x ** 2)) if self.normalize == "rms" else np.abs(x).max()
            if level > 0:
                x = x * (target / level)
            x = np.clip(x, -1.0, 1.0)
        if self.pre_emphasis and x.size:
            x = np.append(x[0], x[1:] - self.pre_emphasis * x[:-1])
        if self.duration is not None:
            n = int(round(self.duration * self.sr))
            x = x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))
        return x.astype(np.float32)

    def transform(self, paths) -> list | np.ndarray:
        """Processed signals: an ``(n, samples)`` array if ``duration`` is set, else a list."""
        out = [self.process(*read_audio(p)) for p in _paths(paths)]
        return np.stack(out) if self.duration is not None and out else out

    def to_dict(self) -> dict:
        return {"sr": self.sr, "duration": self.duration, "trim_db": self.trim_db, "normalize": self.normalize,
                "target_dbfs": self.target_dbfs, "pre_emphasis": self.pre_emphasis}

    def save(self, path) -> Path:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return Path(path)

    @classmethod
    def load(cls, path) -> "AudioPreprocessor":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d.pop("sr"), **d)


def mel_filterbank(sr: int, n_fft: int, n_mels: int = 40, fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    fmax = fmax or sr / 2
    mel = lambda f: 2595 * np.log10(1 + f / 700)          # noqa: E731
    inv = lambda m: 700 * (10 ** (m / 2595) - 1)          # noqa: E731
    points = inv(np.linspace(mel(fmin), mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * points / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        lo, c, hi = bins[m - 1], bins[m], bins[m + 1]
        if c > lo:
            fb[m - 1, lo:c] = (np.arange(lo, c) - lo) / (c - lo)
        if hi > c:
            fb[m - 1, c:hi] = (hi - np.arange(c, hi)) / (hi - c)
    return fb


def spectral_features(x: np.ndarray, sr: int, *, n_fft: int = 512, hop: int = 160, n_mels: int = 40,
                      n_mfcc: int = 13) -> dict:
    """Frame-level features summarised by mean and std over time."""
    from scipy.fft import dct
    if x.size == 0:
        x = np.zeros(n_fft, np.float32)
    frames = _frames(x, n_fft, hop) * np.hanning(n_fft)
    spec = np.abs(np.fft.rfft(frames, axis=1))
    power = spec ** 2
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    mel = np.log(np.maximum(power @ mel_filterbank(sr, n_fft, n_mels).T, 1e-10))
    mfcc = dct(mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]
    delta = np.diff(mfcc, axis=0) if len(mfcc) > 1 else np.zeros_like(mfcc)
    total = spec.sum(axis=1) + 1e-10
    centroid = (spec * freqs).sum(axis=1) / total
    bandwidth = np.sqrt((spec * (freqs - centroid[:, None]) ** 2).sum(axis=1) / total)
    cum = np.cumsum(spec, axis=1)
    rolloff = freqs[np.argmax(cum >= 0.85 * cum[:, -1:], axis=1)]
    flatness = np.exp(np.mean(np.log(spec + 1e-10), axis=1)) / (spec.mean(axis=1) + 1e-10)
    raw = _frames(x, n_fft, hop)
    zcr = np.mean(np.abs(np.diff(np.sign(raw), axis=1)) > 0, axis=1)
    rms = np.sqrt((raw ** 2).mean(axis=1))
    out = {"duration": len(x) / sr}
    for name, series in (("centroid", centroid), ("bandwidth", bandwidth), ("rolloff", rolloff),
                         ("flatness", flatness), ("zcr", zcr), ("rms", rms)):
        out[f"{name}_mean"], out[f"{name}_std"] = float(series.mean()), float(series.std())
    for k in range(mfcc.shape[1]):
        out[f"mfcc_{k}_mean"], out[f"mfcc_{k}_std"] = float(mfcc[:, k].mean()), float(mfcc[:, k].std())
        out[f"mfcc_delta_{k}_std"] = float(delta[:, k].std())
    return out


def audio_features(source, *, preprocessor: AudioPreprocessor | None = None, **kw) -> pl.DataFrame:
    """One row of features per file (``path`` + numeric columns) for ``AutoML``."""
    pre = preprocessor or AudioPreprocessor(sr=16000, duration=None, trim_db=-40.0, normalize="rms")
    rows, skipped = [], []
    for path in _paths(source):
        try:
            x = pre.process(*read_audio(path))
        except Exception as exc:                 # one corrupt file must not sink the whole batch
            skipped.append(f"{path} ({type(exc).__name__})")
            continue
        rows.append({"path": path, **spectral_features(x, pre.sr, **kw)})
    if skipped:
        import warnings

        from ..core import MLPilotWarning
        warnings.warn(f"Skipped {len(skipped)} unreadable audio file(s): {skipped[:5]}", MLPilotWarning,
                      stacklevel=2)
    return pl.DataFrame(rows)
