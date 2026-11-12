"""Weather, lighting and capture conditions: detect them, simulate them, and undo them.

Real-world images are taken in fog, rain, snow, at night, against the sun, from a moving car.
A model trained only on clear daylight images fails silently on the rest. This module:

>>> from mlpilot.media import conditions as cond
>>> cond.detect_conditions("camera_042.jpg")
ConditionResult(tags=['fog', 'low_light'], scores={'fog': 0.71, 'low_light': 0.64, ...})
>>> cond.condition_report("traffic_cams/")          # one row per image + coverage per condition
>>> foggy = cond.apply_condition(img, "fog", severity=4, seed=1)        # training augmentation
>>> clear = cond.enhance(foggy)                                         # dehaze / brighten / denoise

Detected: ``fog`` (haze), ``rain``, ``snow``, ``low_light``, ``glare`` (sun flare, headlights),
``motion_blur`` (any strong blur; the ``anisotropy`` statistic separates motion from focus blur),
``noise``, ``jpeg`` (compression artefacts). ``shadow`` and ``cutout`` can be *simulated* but are
not detected: brightness statistics cannot tell a cast shadow from a dark object. Each
detector returns a 0-1 score; a condition is *tagged* when its score passes a threshold
(``THRESHOLDS``, adjustable). The detectors are classical image statistics: dark-channel haze
estimation, an Immerkaer noise estimate, gradient anisotropy for motion blur, oriented
bright-streak analysis for rain, small bright blobs for snow, 8x8 block-edge energy for JPEG.
They work with Pillow + numpy + scipy alone (no OpenCV, no GPU) and are meant for auditing and
routing data, not as a substitute for a trained weather classifier.
"""

from __future__ import annotations

import dataclasses
import io

import numpy as np
import polars as pl
from scipy import ndimage

from .image import _items, _pil

CONDITIONS = ("fog", "rain", "snow", "low_light", "glare", "motion_blur", "noise", "jpeg")      # detected
SIMULATED = CONDITIONS + ("shadow", "cutout")                                                     # can be applied
THRESHOLDS = {c: 0.5 for c in CONDITIONS}


# ── helpers ─────────────────────────────────────────────────────────────────
def _rgb(image, max_side: int = 512) -> np.ndarray:
    """PIL image, RGB array or path -> float RGB array in [0, 1], downscaled for speed."""
    Image, _ = _pil()
    im = _as_pil(image)
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def _as_pil(image):
    """PIL image, RGB / gray array (uint8 or 0-1 float) or path -> RGB PIL image."""
    Image, _ = _pil()
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        arr = image if image.dtype == np.uint8 else (np.clip(image, 0, 1) * 255 + 0.5).astype(np.uint8)
        arr = np.stack([arr] * 3, axis=2) if arr.ndim == 2 else arr[..., :3]
        return Image.fromarray(np.ascontiguousarray(arr))
    from .image import open_image
    return open_image(image, "RGB")


def _to_pil(arr: np.ndarray):
    Image, _ = _pil()
    return Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8))


def _gray(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _ramp(x: float, lo: float, hi: float) -> float:
    """0 below ``lo``, 1 above ``hi``, linear in between."""
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def dark_channel(rgb: np.ndarray, patch: int = 15) -> np.ndarray:
    """Per-pixel minimum over colour channels and a local patch (He et al.'s haze prior):
    close to 0 in clear outdoor images, high where haze adds a bright veil."""
    return ndimage.minimum_filter(rgb.min(axis=2), size=patch)


def noise_sigma(gray: np.ndarray) -> float:
    """Immerkaer's fast noise estimate (standard deviation of additive noise, 0-1 scale)."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    conv = ndimage.convolve(gray, k, mode="reflect")[1:-1, 1:-1]
    h, w = gray.shape
    return float(np.sqrt(np.pi / 2) * np.abs(conv).sum() / (6 * max(w - 2, 1) * max(h - 2, 1)))


def _directional_energy(gray: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(gray)
    d1, d2 = (gx + gy) / np.sqrt(2), (gx - gy) / np.sqrt(2)
    return np.array([np.mean(gx ** 2), np.mean(d1 ** 2), np.mean(gy ** 2), np.mean(d2 ** 2)]) + 1e-9


# ── detection ───────────────────────────────────────────────────────────────
@dataclasses.dataclass
class ConditionResult:
    scores: dict
    tags: list
    stats: dict

    @property
    def clear(self) -> bool:
        return not self.tags

    def __repr__(self) -> str:
        shown = {k: round(v, 2) for k, v in self.scores.items() if v >= 0.2}
        return f"ConditionResult(tags={self.tags}, scores={shown})"


def _full_crop(image, side: int = 768) -> np.ndarray:
    """A full-resolution centre crop (gray, 0-1). Noise and JPEG blocks must be measured
    without resampling: downscaling averages noise away and smooths block edges."""
    g = np.asarray(_as_pil(image).convert("L"), dtype=np.float32) / 255.0
    h, w = g.shape
    y0, x0 = max(0, (h - side) // 2), max(0, (w - side) // 2)
    return g[y0:y0 + side, x0:x0 + side]


def _streaks_and_flakes(gray: np.ndarray) -> tuple:
    """Rain: many thin bright components pointing the *same* way. Snow: many small round bright
    blobs spread over the *whole* frame. Natural texture (hair, foam, fabric) fails one of the two."""
    tophat = gray - ndimage.grey_opening(gray, size=(5, 5))
    bright = tophat > max(0.06, float(np.percentile(tophat, 99)) * 0.5)
    lbl, n = ndimage.label(bright)
    angles, blobs = [], []
    if n:
        for i, sl in enumerate(ndimage.find_objects(lbl), start=1):
            ys, xs = np.nonzero(lbl[sl] == i)
            if ys.size < 4:
                continue
            cy, cx = ys.mean(), xs.mean()
            mu20, mu02, mu11 = ((xs - cx) ** 2).mean(), ((ys - cy) ** 2).mean(), ((xs - cx) * (ys - cy)).mean()
            common = np.sqrt(((mu20 - mu02) / 2) ** 2 + mu11 ** 2)
            l1, l2 = (mu20 + mu02) / 2 + common, max((mu20 + mu02) / 2 - common, 1e-3)
            length = 4 * np.sqrt(l1)
            if l1 / l2 >= 9 and length >= 8:                     # elongated: a streak candidate
                angles.append(0.5 * np.arctan2(2 * mu11, mu20 - mu02))
            elif 5 <= ys.size <= 45 and l1 / l2 <= 4:             # small, round, bigger than a noise speck
                blobs.append((cy + sl[0].start, cx + sl[1].start))
    area = gray.size / 1e4
    density_streak = len(angles) / area
    concentration = float(np.abs(np.mean(np.exp(2j * np.array(angles))))) if len(angles) >= 5 else 0.0
    density_blob = len(blobs) / area
    if blobs:
        h, w = gray.shape
        cells = np.zeros((4, 4))
        for y, x in blobs:
            cells[min(int(4 * y / h), 3), min(int(4 * x / w), 3)] += 1
        spread = float((cells >= max(2, len(blobs) / 64)).mean())
    else:
        spread = 0.0
    return density_streak, concentration, density_blob, spread


def condition_scores(image) -> tuple:
    """``(scores, stats)``: a 0-1 score per condition, and the raw statistics behind them."""
    rgb = _rgb(image)
    gray = _gray(rgb)
    crop = _full_crop(image)
    mx, mn = rgb.max(axis=2), rgb.min(axis=2)
    sat = np.where(mx > 1e-3, (mx - mn) / np.maximum(mx, 1e-3), 0)
    dc = dark_channel(rgb)
    lum_mean, lum_p95 = float(gray.mean()), float(np.percentile(gray, 95))
    contrast = float(gray.std())
    sigma = noise_sigma(crop)
    sharp = float(ndimage.laplace(gray).var())
    rel_sharp = sharp / (float(gray.var()) + 1e-6)          # contrast-normalised: fog / darkness aren't blur
    energy = _directional_energy(ndimage.gaussian_filter(gray, 0.7))
    anisotropy = float(energy.max() / energy.min())
    streak_density, concentration, blob_density, spread = _streaks_and_flakes(gray)
    glare_frac = float((gray > 0.96).mean())
    # jpeg blockiness: luminance jumps on the 8-pixel grid vs elsewhere (full resolution)
    if crop.shape[0] > 32 and crop.shape[1] > 32:
        dh = np.abs(np.diff(crop, axis=1))
        blockiness = float(dh[:, 7::8].mean() / (np.delete(dh, np.s_[7::8], axis=1).mean() + 1e-6))
    else:
        blockiness = 1.0

    stats = {"brightness": lum_mean, "p95_brightness": lum_p95, "contrast": contrast, "saturation": float(sat.mean()),
             "dark_channel": float(dc.mean()), "noise_sigma": sigma, "sharpness": sharp,
             "relative_sharpness": rel_sharp, "anisotropy": anisotropy,
             "streaks_per_100px2": streak_density, "streak_alignment": concentration,
             "flakes_per_100px2": blob_density, "flake_spread": spread, "glare_fraction": glare_frac,
             "blockiness": blockiness}
    blur = _ramp(0.06 - rel_sharp, 0.0, 0.04)
    scores = {
        "fog": min(_ramp(float(dc.mean()), 0.25, 0.5), _ramp(0.35 - float(sat.mean()), 0.05, 0.2) + 0.5,
                   _ramp(0.22 - contrast, 0.0, 0.1) + 0.5),
        # aligned streaks on a still-sharp scene (motion blur also aligns highlights, but blurs everything)
        "rain": _ramp(streak_density, 0.4, 1.5) * _ramp(concentration, 0.55, 0.85) * (1 - blur),
        # many small bright flakes across the whole frame; very strong noise also makes specks
        "snow": _ramp(blob_density, 6.0, 12.0) * _ramp(spread, 0.6, 0.9) * (1 - _ramp(sigma, 0.03, 0.045)),
        "low_light": _ramp(0.32 - lum_mean, 0.0, 0.17) * _ramp(0.75 - lum_p95, 0.0, 0.3),
        "glare": _ramp(glare_frac, 0.02, 0.1),
        "motion_blur": blur,                                     # strong blur (motion or focus); see anisotropy
        "noise": _ramp(sigma, 0.012, 0.03),
        "jpeg": _ramp(blockiness, 1.15, 1.5),
    }
    return scores, stats


def detect_conditions(image, *, thresholds: dict | None = None) -> ConditionResult:
    """Score and tag the conditions an image was taken in (see the module docstring)."""
    th = {**THRESHOLDS, **(thresholds or {})}
    scores, stats = condition_scores(image)
    return ConditionResult(scores, [c for c in CONDITIONS if scores[c] >= th[c]], stats)


def condition_report(source, *, thresholds: dict | None = None, min_share: float = 0.05) -> tuple:
    """``(per_image, coverage)`` for a folder / glob / list of paths or PIL images.

    ``coverage`` counts images per condition and flags conditions that are present but rarer
    than ``min_share`` of the data, and conditions with no examples at all: a model will not
    learn to handle what it has (almost) never seen."""
    rows = []
    for name, im in _items(source):
        try:
            r = detect_conditions(im if im is not None else name, thresholds=thresholds)
        except Exception as exc:
            rows.append({"path": name, "tags": None, "error": f"{type(exc).__name__}: {exc}"[:200]})
            continue
        rows.append({"path": name, "tags": ",".join(r.tags) or "clear", "error": None,
                     **{f"score_{k}": round(v, 4) for k, v in r.scores.items()}})
    per_image = pl.DataFrame(rows, infer_schema_length=None)
    n = max(per_image.filter(pl.col("error").is_null()).height, 1)
    cov = []
    for c in ("clear",) + CONDITIONS:
        k = int(per_image.filter(pl.col("tags").fill_null("").str.split(",").list.contains(c)).height)
        status = "ok" if k / n >= min_share else ("missing" if k == 0 else "rare")
        cov.append({"condition": c, "images": k, "share": round(k / n, 4),
                    "status": "ok" if c == "clear" else status})
    return per_image, pl.DataFrame(cov)


# ── simulation (augmentation) ───────────────────────────────────────────────
def _severity(severity) -> float:
    s = float(severity)
    if not 1 <= s <= 5:
        raise ValueError("severity runs from 1 (mild) to 5 (extreme)")
    return s


def apply_condition(image, condition: str, *, severity: float = 3, seed: int = 0):
    """A copy of ``image`` (PIL) as if taken in ``condition`` at ``severity`` 1-5. Reproducible
    for a given ``seed``. Use it to augment training data or to stress-test a model
    (``media.benchmark.condition_benchmark``)."""
    Image, _ = _pil()
    from PIL import ImageDraw, ImageFilter
    s = _severity(severity)
    rng = np.random.default_rng(seed)
    base = _as_pil(image)
    arr = np.asarray(base, dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    if condition == "fog":
        depth = np.linspace(1.0, 0.35, h, dtype=np.float32)[:, None, None]          # the top is further away
        texture = ndimage.gaussian_filter(rng.normal(0, 1, (h, w)).astype(np.float32), sigma=max(h, w) / 25)
        texture = (texture - texture.min()) / (np.ptp(texture) + 1e-6)
        t = np.exp(-(0.35 + 0.33 * s) * depth * (0.8 + 0.4 * texture[..., None]))
        airlight = 0.78 + 0.04 * s
        out = arr * t + airlight * (1 - t)
    elif condition == "rain":
        im = base.copy()
        layer = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(layer)
        n = int(w * h / 1000 * (0.6 + 0.6 * s))
        angle = np.radians(rng.uniform(-15, 15))
        for _ in range(n):
            x, y = rng.uniform(0, w), rng.uniform(0, h)
            ln = rng.uniform(10, 18 + 4 * s)
            d.line([(x, y), (x + ln * np.sin(angle), y + ln * np.cos(angle))], fill=int(rng.uniform(150, 230)), width=1)
        layer = layer.filter(ImageFilter.GaussianBlur(0.5))
        streaks = np.asarray(layer, dtype=np.float32)[..., None] / 255.0
        dim = arr * (1 - 0.04 * s)
        out = dim + streaks * (0.55 + 0.05 * s) * (1 - dim)
        out = np.asarray(_to_pil(out).filter(ImageFilter.GaussianBlur(0.3 + 0.05 * s)), dtype=np.float32) / 255.0
    elif condition == "snow":
        layer = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(layer)
        n = int(w * h / 600 * (0.5 + 0.7 * s))
        for _ in range(n):
            x, y, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0.8, 1.6 + 0.3 * s)
            d.ellipse([x - r, y - r, x + r, y + r], fill=int(rng.uniform(200, 255)))
        flakes = np.asarray(layer.filter(ImageFilter.GaussianBlur(0.4)), dtype=np.float32)[..., None] / 255.0
        whiten = arr * (1 - 0.05 * s) + 0.05 * s * 0.85
        out = whiten + flakes * (1 - whiten)
    elif condition == "low_light":
        dark = np.power(arr, 1.3 + 0.35 * s) * (0.75 - 0.1 * s)
        noise = rng.normal(0, 0.006 + 0.004 * s, dark.shape).astype(np.float32)
        out = dark + noise
    elif condition == "glare":
        cy, cx = rng.uniform(0.1, 0.5) * h, rng.uniform(0.2, 0.8) * w
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.hypot(yy - cy, xx - cx) / max(h, w)
        flare = np.exp(-(r / (0.06 + 0.04 * s)) ** 2)[..., None]
        bloom = np.exp(-(r / (0.25 + 0.05 * s)) ** 2)[..., None] * (0.15 + 0.05 * s)
        out = arr + (1 - arr) * np.clip(flare * 1.2 + bloom, 0, 1)
    elif condition == "shadow":
        mask = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(mask)
        pts = [(rng.uniform(-0.1, 0.6) * w, rng.uniform(-0.1, 1.1) * h) for _ in range(5)]
        pts.sort(key=lambda p: np.arctan2(p[1] - h / 2, p[0] - w / 2))
        d.polygon(pts, fill=255)
        m = np.asarray(mask.filter(ImageFilter.GaussianBlur(2)), dtype=np.float32)[..., None] / 255.0
        out = arr * (1 - m * (0.25 + 0.11 * s))
    elif condition == "motion_blur":
        length = int(3 + 4 * s)
        angle = rng.uniform(0, 180)
        k = np.zeros((length, length), dtype=np.float32)
        k[length // 2, :] = 1.0
        k = ndimage.rotate(k, angle, reshape=False, order=1)
        k /= k.sum()
        out = np.stack([ndimage.convolve(arr[..., c], k, mode="reflect") for c in range(3)], axis=2)
    elif condition == "noise":
        out = arr + rng.normal(0, 0.015 + 0.015 * s, arr.shape).astype(np.float32)
    elif condition == "jpeg":
        buf = io.BytesIO()
        base.save(buf, format="JPEG", quality=int(max(4, 40 - 7 * s)))
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    elif condition == "cutout":
        out = arr.copy()
        frac = 0.06 * s
        ch, cw = int(h * np.sqrt(frac)), int(w * np.sqrt(frac))
        y0, x0 = int(rng.uniform(0, h - ch)), int(rng.uniform(0, w - cw))
        out[y0:y0 + ch, x0:x0 + cw] = rng.uniform(0, 1, 3)
    else:
        raise ValueError(f"Unknown condition {condition!r}; choose from {SIMULATED}")
    return _to_pil(out)


# ── restoration ─────────────────────────────────────────────────────────────
def _guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    size = 2 * r + 1
    mean = lambda x: ndimage.uniform_filter(x, size=size, mode="reflect")   # noqa: E731
    mi, ms = mean(guide), mean(src)
    cov = mean(guide * src) - mi * ms
    var = mean(guide * guide) - mi * mi
    a = cov / (var + eps)
    b = ms - a * mi
    return mean(a) * guide + mean(b)


def dehaze(image, *, strength: float = 0.9):
    """Remove fog / haze with the dark-channel prior and a guided-filter transmission map."""
    arr = np.asarray(_as_pil(image), dtype=np.float32) / 255.0
    dc = dark_channel(arr)
    n = max(1, int(dc.size * 0.001))
    idx = np.argsort(dc.ravel())[-n:]
    airlight = arr.reshape(-1, 3)[idx].mean(axis=0)
    t = 1 - strength * dark_channel(arr / np.maximum(airlight, 1e-3))
    t = _guided_filter(_gray(arr), t, r=max(8, min(arr.shape[:2]) // 25), eps=1e-3)
    t = np.clip(t, 0.12, 1.0)[..., None]
    out = (arr - airlight) / t + airlight
    lo, hi = np.percentile(out, 0.5), np.percentile(out, 99.5)                 # gentle contrast stretch
    return _to_pil((out - lo) / max(hi - lo, 1e-3))


def brighten(image, *, target: float = 0.45):
    """Low-light enhancement: adaptive gamma towards ``target`` mean brightness, then CLAHE when
    OpenCV is available (local contrast without blowing out highlights)."""
    arr = np.asarray(_as_pil(image), dtype=np.float32) / 255.0
    mean = float(_gray(arr).mean())
    gamma = np.clip(np.log(target) / np.log(max(mean, 1e-3)), 0.25, 1.0)
    out = np.power(np.clip(arr, 0, 1), gamma)
    try:
        import cv2
        lab = cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
        return _pil()[0].fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
    except ImportError:
        return _to_pil(out)


def denoise(image, *, strength: float | None = None):
    """Reduce sensor noise: non-local means with OpenCV, otherwise an edge-preserving median."""
    arr8 = np.asarray(_as_pil(image))
    sigma = noise_sigma(_gray(arr8.astype(np.float32) / 255.0))
    h = strength if strength is not None else float(np.clip(sigma * 255 * 0.9, 3, 25))
    try:
        import cv2
        return _pil()[0].fromarray(cv2.fastNlMeansDenoisingColored(arr8, None, h, h, 7, 21))
    except ImportError:
        return _pil()[0].fromarray(np.stack([ndimage.median_filter(arr8[..., c], size=3) for c in range(3)], axis=2))


def sharpen(image, *, amount: float = 1.0, radius: float = 2.0):
    """Unsharp mask: recovers edge detail lost to mild blur (motion or focus)."""
    from PIL import ImageFilter
    return _as_pil(image).filter(ImageFilter.UnsharpMask(radius=radius, percent=int(150 * amount), threshold=2))


_STEPS = {"dehaze": dehaze, "brighten": brighten, "denoise": denoise, "sharpen": sharpen}
_AUTO = {"fog": "dehaze", "low_light": "brighten", "noise": "denoise", "motion_blur": "sharpen"}


def enhance(image, steps="auto"):
    """Restore an image. ``steps="auto"`` picks from the detected conditions (fog -> dehaze,
    low light -> brighten, noise -> denoise, blur -> sharpen); or pass a list such as
    ``["dehaze", "denoise"]``. Returns ``(image, steps_applied)``."""
    im = _as_pil(image)
    if steps == "auto":
        tags = detect_conditions(im).tags
        order = ["denoise", "dehaze", "brighten", "sharpen"]
        steps = [s for s in order if s in {_AUTO[t] for t in tags if t in _AUTO}]
    unknown = [s for s in steps if s not in _STEPS]
    if unknown:
        raise ValueError(f"Unknown steps {unknown}; choose from {list(_STEPS)}")
    for s in steps:
        im = _STEPS[s](im)
    return im, list(steps)
