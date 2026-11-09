"""Image and audio preprocessing, quality audits and features."""

import numpy as np
import polars as pl
import pytest
from scipy.io import wavfile

import automatedcleaning as ac
from automatedcleaning.media import (AudioPreprocessor, ImagePreprocessor, audio_features, audio_quality_report, augment,
                                     image_features, image_quality_report, load_audio, scan_audio, scan_images)

try:
    from PIL import Image, ImageFilter
except ImportError:                       # image tests skip; audio tests don't need Pillow
    Image = ImageFilter = None
needs_pil = pytest.mark.skipif(Image is None, reason="needs Pillow ([image] extra)")


def _stripes(vertical: bool, seed: int, size=48):
    rng = np.random.default_rng(seed)
    a = np.zeros((size, size, 3), np.uint8)
    period = int(rng.integers(4, 8))
    for i in range(0, size, period * 2):
        if vertical:
            a[:, i:i + period] = 220
        else:
            a[i:i + period, :] = 220
    a = np.clip(a.astype(int) + rng.integers(-20, 20, a.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(a)


@pytest.fixture(scope="module")
def image_dir(tmp_path_factory):
    if Image is None:
        pytest.skip("needs Pillow")
    d = tmp_path_factory.mktemp("imgs")
    for cls, vertical in (("vertical", True), ("horizontal", False)):
        (d / cls).mkdir()
        for i in range(30):
            _stripes(vertical, i + (100 if vertical else 0)).save(d / cls / f"{i}.png")
    odd = d / "odd"
    odd.mkdir()
    (odd / "broken.jpg").write_bytes(b"\xff\xd8\xff not really a jpeg")
    base = _stripes(True, 1)
    base.save(odd / "copy.png")                                         # byte copy below
    (odd / "copy2.png").write_bytes((odd / "copy.png").read_bytes())
    base.resize((96, 96)).save(odd / "resized.jpg", quality=90)         # near duplicate
    Image.new("RGB", (16, 16), (128, 128, 128)).save(odd / "tiny.png")
    Image.new("RGB", (64, 64), (3, 3, 3)).save(odd / "dark.png")
    _stripes(False, 5, 128).filter(ImageFilter.GaussianBlur(12)).save(odd / "blurry.png")
    Image.new("RGBA", (64, 64), (200, 10, 10, 255)).save(odd / "rgba.png")
    return d


@needs_pil
def test_scan_and_quality_report(image_dir):
    odd = scan_images(image_dir / "odd")
    assert odd.height == 8 and odd.filter(pl.col("error").is_not_null())["path"].str.contains("broken").all()
    rep = image_quality_report(scan_images(image_dir))           # blur is judged against the whole set
    checks = {i.check: i for i in rep.issues}
    for c in ("unreadable", "exact_duplicates", "near_duplicates", "too_small", "too_dark", "blurry", "mixed_color_modes"):
        assert c in checks, c
    assert any("resized" in f for f in checks["near_duplicates"].files)
    assert any("blurry" in f for f in checks["blurry"].files)
    assert "broken.jpg" in " ".join(rep.problem_files)
    labelled = scan_images(image_dir / "*" / "*.png", label_from_folder=True)
    assert set(labelled["label"].drop_nulls().unique()) >= {"vertical", "horizontal", "odd"}


@needs_pil
def test_exif_orientation_is_applied(tmp_path):
    im = Image.new("RGB", (40, 20), (10, 200, 10))
    exif = im.getexif()
    exif[0x0112] = 6                                     # "rotate 90 CW" - typical phone photo
    im.save(tmp_path / "phone.jpg", exif=exif)
    assert ac.media.open_image(tmp_path / "phone.jpg").size == (20, 40)


@needs_pil
def test_preprocessor_shapes_normalisation_and_roundtrip(image_dir, tmp_path):
    paths = sorted(str(p) for p in (image_dir / "vertical").glob("*.png"))[:10] + [str(image_dir / "odd" / "rgba.png")]
    pre = ImagePreprocessor((32, 24), resize="pad")
    X = pre.fit_transform(paths)
    assert X.shape == (11, 24, 32, 3) and X.dtype == np.float32
    assert np.allclose(X.mean(axis=(0, 1, 2)), 0, atol=1e-4) and np.allclose(X.std(axis=(0, 1, 2)), 1, atol=1e-3)
    gray = ImagePreprocessor((16, 16), mode="L", resize="crop", normalize=False).transform(paths[:2])
    assert gray.shape == (2, 16, 16, 1) and gray.max() <= 1.0
    loaded = ImagePreprocessor.load(pre.save(tmp_path / "pre.json"))
    assert np.allclose(loaded.transform(paths[:3]), X[:3])
    with pytest.raises(RuntimeError, match="fit"):
        ImagePreprocessor((8, 8)).transform(paths[:1])


@needs_pil
def test_augment_is_reproducible(image_dir):
    im = Image.open(next((image_dir / "vertical").glob("*.png"))).convert("RGB")
    a, b, c = augment(im, seed=7), augment(im, seed=7), augment(im, seed=8)
    assert np.array_equal(np.asarray(a), np.asarray(b)) and not np.array_equal(np.asarray(a), np.asarray(c))
    assert a.size == im.size


@needs_pil
def test_image_features_train_an_automl_classifier(image_dir):
    feats = image_features(image_dir / "*" / "*.png").filter(~pl.col("path").str.contains("odd"))
    labels = scan_images(image_dir / "*" / "*.png", label_from_folder=True).select("path", "label")
    df = feats.join(labels, on="path").drop("path").sample(fraction=1.0, shuffle=True, seed=0)
    assert df.width > 60 and df.height == 60
    model = ac.AutoML("label", time_budget=20, cv=3, verbose=False, max_trials=1,
                      include=["logistic_regression"]).fit(df.head(40))
    assert model.evaluate(df.tail(20))["accuracy"] >= 0.9


# ── audio ───────────────────────────────────────────────────────────────────

def _tone(path, freq, sr=16000, seconds=1.0, amp=0.3, noise=0.01, seed=0, lead_silence=0.0, stereo=False, dc=0.0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds)) / sr
    x = amp * np.sin(2 * np.pi * freq * t) + noise * rng.normal(size=t.size) + dc
    x = np.concatenate([np.zeros(int(sr * lead_silence)), x])
    data = np.stack([x, x], axis=1) if stereo else x
    wavfile.write(path, sr, (np.clip(data, -1, 1) * 32767).astype(np.int16))


@pytest.fixture(scope="module")
def audio_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("audio")
    for cls, freq in (("low", 300), ("high", 2500)):
        (d / cls).mkdir()
        for i in range(20):
            _tone(d / cls / f"{i}.wav", freq * (1 + 0.05 * (i % 5)), seed=i)
    bad = d / "bad"
    bad.mkdir()
    wavfile.write(bad / "silent.wav", 16000, np.zeros(16000, np.int16))
    _tone(bad / "clipped.wav", 440, amp=1.6)
    _tone(bad / "short.wav", 440, seconds=0.1)
    _tone(bad / "stereo44k.wav", 440, sr=44100, stereo=True)
    _tone(bad / "dc.wav", 440, dc=0.2, amp=0.1)
    _tone(bad / "quiet_start.wav", 440, lead_silence=6.0)
    (bad / "corrupt.wav").write_bytes(b"RIFF....WAVEjunk")
    return d


def test_audio_scan_and_quality(audio_dir):
    scan = scan_audio(audio_dir / "bad")
    assert scan.filter(pl.col("path").str.contains("stereo44k"))["channels"][0] == 2
    rep = audio_quality_report(scan)
    checks = {i.check: i for i in rep.issues}
    for c in ("unreadable", "silent", "clipping", "too_short", "dc_offset", "mostly_silence", "mixed_sample_rates"):
        assert c in checks, c
    assert any("corrupt" in f for f in checks["unreadable"].files)
    assert any("quiet_start" in f for f in checks["mostly_silence"].files)


def test_audio_preprocessor(audio_dir, tmp_path):
    x, sr = load_audio(audio_dir / "bad" / "stereo44k.wav", sr=16000)
    assert sr == 16000 and x.ndim == 1 and abs(len(x) - 16000) <= 2
    pre = AudioPreprocessor(16000, duration=2.0, normalize="rms", target_dbfs=-20)
    out = pre.transform([str(audio_dir / "bad" / "quiet_start.wav"), str(audio_dir / "low" / "0.wav")])
    assert out.shape == (2, 32000)
    sig = pre.process(*ac.media.read_audio(audio_dir / "bad" / "quiet_start.wav"))
    voiced = sig[: int(0.9 * 16000)]
    assert 20 * np.log10(np.sqrt(np.mean(voiced ** 2))) == pytest.approx(-20, abs=1.0)    # silence trimmed, level set
    assert AudioPreprocessor.load(pre.save(tmp_path / "a.json")).to_dict() == pre.to_dict()


def test_audio_features_separate_classes(audio_dir):
    feats = audio_features(audio_dir / "*" / "*.wav")
    feats = feats.filter(~pl.col("path").str.contains("bad"))
    lab = feats.select("path", pl.col("path").str.contains("high").cast(pl.Int8).alias("is_high"))
    low = feats.filter(~pl.col("path").str.contains("high"))["centroid_mean"].mean()
    high = feats.filter(pl.col("path").str.contains("high"))["centroid_mean"].mean()
    assert high > 2 * low                     # magnitude-weighted centroid: noise pulls both toward mid-band
    df = feats.join(lab, on="path").drop("path").sample(fraction=1.0, shuffle=True, seed=1)
    assert any(c.startswith("mfcc_") for c in df.columns)
    model = ac.AutoML("is_high", time_budget=20, cv=3, verbose=False, max_trials=1,
                      include=["logistic_regression"]).fit(df.head(28))
    assert model.evaluate(df.tail(12))["accuracy"] == 1.0
