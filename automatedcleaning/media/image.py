"""Image data: audit a folder of images, preprocess them consistently, and turn them into
features the tabular AutoML can learn from (no deep-learning dependency).

Needs Pillow: ``pip install "AutomatedCleaning[image]"``.

* ``scan_images("photos/**/*.jpg")``       one row per file: size, mode, format, hashes, brightness,
                                             sharpness, readable or not
* ``image_quality_report(scan)``           corrupt files, exact & near duplicates, tiny / dark /
                                             over-exposed / blurry images, odd aspect ratios, mixed
                                             colour modes, label imbalance
* ``ImagePreprocessor(size=(224, 224))``   EXIF-orientation fix, colour mode, resize (pad / crop /
                                             stretch), per-channel normalisation learned in ``fit``;
                                             JSON-serialisable like every pipeline step
* ``image_features(paths)``                colour histograms, edge-orientation histograms (HOG-like),
                                             texture and shape statistics -> a DataFrame for AutoML
* ``augment(image, seed=…)``               reproducible flips, rotations, crops, brightness/contrast
"""

from __future__ import annotations

import dataclasses
import glob as _glob
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp")


def _pil():
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise ImportError('Image support needs Pillow: pip install "AutomatedCleaning[image]"') from exc
    return Image, ImageOps


def _paths(source) -> list:
    if isinstance(source, (str, Path)):
        s = str(source)
        if Path(s).is_dir():
            return sorted(str(p) for p in Path(s).rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        return sorted(p for p in _glob.glob(s, recursive=True) if Path(p).suffix.lower() in IMAGE_EXTS)
    return [str(p) for p in source]


def open_image(path, mode: str | None = "RGB"):
    """Open an image with its EXIF orientation applied (phone photos are often stored rotated)."""
    Image, ImageOps = _pil()
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im.load()
        return im.convert(mode) if mode else im.copy()


def dhash(image, size: int = 8) -> str:
    """Perceptual difference hash: near-identical images (resized, re-compressed) share most bits."""
    Image, _ = _pil()
    g = np.asarray(image.convert("L").resize((size + 1, size), Image.Resampling.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):0{size * size // 4}x}"


def _hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian: low values mean a blurry image."""
    g = gray.astype(np.float32)
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var()) if lap.size else 0.0


def scan_images(source, *, labels: dict | None = None, label_from_folder: bool = False) -> pl.DataFrame:
    """Inspect every image (files that can't be read are kept, with ``error`` set).

    ``label_from_folder=True`` takes each image's label from its parent folder name
    (the usual ``train/cat/001.jpg`` layout); or pass ``labels={path: label}``."""
    rows = []
    for path in _paths(source):
        row = {"path": path, "label": None, "error": None, "width": None, "height": None, "mode": None,
               "format": None, "bytes": None, "sha256": None, "dhash": None, "brightness": None,
               "contrast": None, "sharpness": None}
        if labels is not None:
            row["label"] = labels.get(path)
        elif label_from_folder:
            row["label"] = Path(path).parent.name
        try:
            data = Path(path).read_bytes()
            row["bytes"], row["sha256"] = len(data), hashlib.sha256(data).hexdigest()
            Image, _ = _pil()
            with Image.open(path) as raw:
                row["format"], row["mode"] = raw.format, raw.mode
            im = open_image(path, mode=None)
            row["width"], row["height"] = im.size
            gray = np.asarray(im.convert("L"), dtype=np.float32)
            row["brightness"], row["contrast"] = float(gray.mean()), float(gray.std())
            row["sharpness"] = sharpness(gray[:: max(1, gray.shape[0] // 512), :: max(1, gray.shape[1] // 512)])
            row["dhash"] = dhash(im)
        except Exception as exc:                       # corrupt / truncated / not an image
            row["error"] = f"{type(exc).__name__}: {exc}"[:200]
        rows.append(row)
    schema = {"path": pl.String, "label": pl.String, "error": pl.String, "width": pl.Int64, "height": pl.Int64,
              "mode": pl.String, "format": pl.String, "bytes": pl.Int64, "sha256": pl.String, "dhash": pl.String,
              "brightness": pl.Float64, "contrast": pl.Float64, "sharpness": pl.Float64}
    return pl.DataFrame(rows, schema=schema)


@dataclasses.dataclass
class MediaIssue:
    check: str
    severity: str
    message: str
    files: list


@dataclasses.dataclass
class MediaQualityReport:
    kind: str
    n_files: int
    issues: list
    summary_stats: dict

    @property
    def problem_files(self) -> list:
        return sorted({f for i in self.issues for f in i.files})

    def summary(self) -> str:
        lines = [f"{self.kind.title()} quality: {self.n_files} files, {len(self.issues)} issue type(s), "
                 f"{len(self.problem_files)} file(s) affected"]
        for i in self.issues:
            lines.append(f"  [{i.severity}] {i.check}: {i.message}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "n_files": self.n_files, "summary_stats": self.summary_stats,
                "issues": [dataclasses.asdict(i) for i in self.issues]}

    def __repr__(self) -> str:
        return self.summary()


def image_quality_report(scan: pl.DataFrame, *, min_side: int = 32, dark: float = 20, bright: float = 235,
                         blur_ratio: float = 0.1, near_duplicate_bits: int = 4, max_aspect: float = 4.0) -> MediaQualityReport:
    """Problems that hurt image models, from a :func:`scan_images` result."""
    issues = []
    bad = scan.filter(pl.col("error").is_not_null())
    if bad.height:
        issues.append(MediaIssue("unreadable", "high", f"{bad.height} file(s) can't be decoded", bad["path"].to_list()))
    ok = scan.filter(pl.col("error").is_null())
    dup = ok.filter(pl.col("sha256").is_duplicated()).sort("sha256")
    if dup.height:
        issues.append(MediaIssue("exact_duplicates", "medium",
                                 f"{dup.height} files are byte-identical copies ({dup['sha256'].n_unique()} groups)",
                                 dup["path"].to_list()))
    near = []
    hashes = [(p, h, s) for p, h, s in ok.select("path", "dhash", "sha256").iter_rows()]
    for i in range(len(hashes)):
        for j in range(i + 1, min(len(hashes), i + 2000)):
            if hashes[i][2] != hashes[j][2] and _hamming(hashes[i][1], hashes[j][1]) <= near_duplicate_bits:
                near += [hashes[i][0], hashes[j][0]]
    if near:
        issues.append(MediaIssue("near_duplicates", "medium",
                                 f"{len(set(near))} files look like resized / re-encoded copies of another",
                                 sorted(set(near))))
    small = ok.filter((pl.col("width") < min_side) | (pl.col("height") < min_side))
    if small.height:
        issues.append(MediaIssue("too_small", "medium", f"{small.height} image(s) smaller than {min_side}px",
                                 small["path"].to_list()))
    for check, cond, text in (("too_dark", pl.col("brightness") < dark, "nearly black"),
                              ("over_exposed", pl.col("brightness") > bright, "nearly white")):
        sel = ok.filter(cond)
        if sel.height:
            issues.append(MediaIssue(check, "low", f"{sel.height} image(s) are {text}", sel["path"].to_list()))
    if ok.height >= 5:
        med = float(ok["sharpness"].median() or 0.0)  # type: ignore[arg-type]
        blurry = ok.filter(pl.col("sharpness") < blur_ratio * med)
        if blurry.height:
            issues.append(MediaIssue("blurry", "low", f"{blurry.height} image(s) far blurrier than typical "
                                                      f"(sharpness < {blur_ratio:.0%} of the median)",
                                     blurry["path"].to_list()))
    odd = ok.filter(((pl.col("width") / pl.col("height")) > max_aspect) | ((pl.col("height") / pl.col("width")) > max_aspect))
    if odd.height:
        issues.append(MediaIssue("extreme_aspect_ratio", "low", f"{odd.height} image(s) wider/taller than {max_aspect}:1",
                                 odd["path"].to_list()))
    modes = ok["mode"].value_counts(sort=True)
    if modes.height > 1:
        issues.append(MediaIssue("mixed_color_modes", "low",
                                 "mixed colour modes " + ", ".join(f"{m} x{n}" for m, n in modes.iter_rows())
                                 + " - convert to one mode (ImagePreprocessor does)", []))
    labels = ok.filter(pl.col("label").is_not_null())["label"].value_counts(sort=True)
    if labels.height >= 2:
        counts = labels["count"].to_list()
        if counts[0] >= 5 * counts[-1]:
            issues.append(MediaIssue("label_imbalance", "medium",
                                     f"most common label has {counts[0]} images, rarest {counts[-1]} "
                                     f"('{labels[labels.columns[0]][-1]}')", []))
    stats = {"readable": ok.height, "median_width": ok["width"].median(), "median_height": ok["height"].median(),
             "formats": dict(ok["format"].value_counts().iter_rows()) if ok.height else {}}
    return MediaQualityReport("image", scan.height, issues, stats)


class ImagePreprocessor:
    """Consistent image preparation, learned on training images and replayed on new ones.

    ``resize``: ``"pad"`` (keep aspect, letterbox), ``"crop"`` (keep aspect, centre crop) or
    ``"stretch"``. ``normalize=True`` learns per-channel mean/std in :meth:`fit`."""

    def __init__(self, size=(224, 224), *, mode: str = "RGB", resize: str = "pad", normalize: bool = True,
                 pad_color=(0, 0, 0)):
        if resize not in ("pad", "crop", "stretch"):
            raise ValueError("resize must be 'pad', 'crop' or 'stretch'")
        self.size, self.mode, self.resize, self.normalize = tuple(size), mode, resize, normalize
        self.pad_color = tuple(pad_color)
        self.mean_ = self.std_ = None

    def _prepare(self, path):
        Image, ImageOps = _pil()
        im = open_image(path, self.mode)
        if self.resize == "stretch":
            im = im.resize(self.size, Image.Resampling.BILINEAR)
        elif self.resize == "crop":
            im = ImageOps.fit(im, self.size, Image.Resampling.BILINEAR)
        else:
            fill = self.pad_color if self.mode != "L" else self.pad_color[0]
            im = ImageOps.pad(im, self.size, Image.Resampling.BILINEAR, color=fill)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        return arr[..., None] if arr.ndim == 2 else arr

    def fit(self, paths) -> "ImagePreprocessor":
        if self.normalize:
            total: np.ndarray = np.zeros(1)
            sq: np.ndarray = np.zeros(1)
            n = 0
            for p in _paths(paths):
                a = self._prepare(p)
                total = total + a.sum(axis=(0, 1))
                sq = sq + (a ** 2).sum(axis=(0, 1))
                n += a.shape[0] * a.shape[1]
            if n == 0:
                raise ValueError("No readable images to fit on")
            self.mean_ = (total / n).astype(float).tolist()
            self.std_ = np.sqrt(np.maximum(sq / n - (total / n) ** 2, 1e-12)).astype(float).tolist()
        return self

    def transform(self, paths) -> np.ndarray:
        """A float32 array ``(n, height, width, channels)``."""
        arrs = [self._prepare(p) for p in _paths(paths)]
        out = np.stack(arrs) if arrs else np.zeros((0, self.size[1], self.size[0], 1 if self.mode == "L" else 3), np.float32)
        if self.normalize:
            if self.mean_ is None:
                raise RuntimeError("Call fit() first (normalize=True learns the channel statistics)")
            out = (out - np.asarray(self.mean_, np.float32)) / np.asarray(self.std_, np.float32)
        return out.astype(np.float32)

    def fit_transform(self, paths) -> np.ndarray:
        return self.fit(paths).transform(paths)

    def to_dict(self) -> dict:
        return {"size": list(self.size), "mode": self.mode, "resize": self.resize, "normalize": self.normalize,
                "pad_color": list(self.pad_color), "mean": self.mean_, "std": self.std_}

    def save(self, path) -> Path:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return Path(path)

    @classmethod
    def load(cls, path) -> "ImagePreprocessor":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(d["size"], mode=d["mode"], resize=d["resize"], normalize=d["normalize"], pad_color=d["pad_color"])
        obj.mean_, obj.std_ = d["mean"], d["std"]
        return obj


def _hog(gray: np.ndarray, cells: int = 4, bins: int = 9) -> np.ndarray:
    gy, gx = np.gradient(gray)
    mag = np.hypot(gx, gy)
    ang = (np.degrees(np.arctan2(gy, gx)) % 180.0)
    h, w = gray.shape
    feats = []
    for i in range(cells):
        for j in range(cells):
            sl = (slice(i * h // cells, (i + 1) * h // cells), slice(j * w // cells, (j + 1) * w // cells))
            hist, _ = np.histogram(ang[sl], bins=bins, range=(0, 180), weights=mag[sl])
            feats.append(hist / (np.linalg.norm(hist) + 1e-6))
    return np.concatenate(feats)


def image_features(source, *, size: int = 64, color_bins: int = 8, hog_cells: int = 4) -> pl.DataFrame:
    """Hand-crafted features per image (``path`` + numeric columns) so ``AutoML`` can classify
    images without a GPU: per-channel colour histograms and moments, HSV saturation, edge
    density, an edge-orientation (HOG-like) grid, sharpness, brightness and aspect ratio."""
    Image, ImageOps = _pil()
    rows, skipped = [], []
    for path in _paths(source):
        try:
            im = open_image(path, "RGB")
        except Exception as exc:                 # one corrupt file must not sink the whole batch
            skipped.append(f"{path} ({type(exc).__name__})")
            continue
        w, h = im.size
        small = np.asarray(ImageOps.fit(im, (size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        gray = small @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        row: dict = {"path": path, "aspect_ratio": w / h, "brightness": float(gray.mean()), "contrast": float(gray.std()),
                     "sharpness": sharpness(gray * 255)}
        for c, name in enumerate("rgb"):
            ch = small[..., c]
            row[f"{name}_mean"], row[f"{name}_std"] = float(ch.mean()), float(ch.std())
            hist, _ = np.histogram(ch, bins=color_bins, range=(0, 1))
            for k, v in enumerate(hist / hist.sum()):
                row[f"{name}_hist_{k}"] = float(v)
        mx, mn = small.max(axis=2), small.min(axis=2)
        row["saturation_mean"] = float(np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0).mean())
        gy, gx = np.gradient(gray)
        row["edge_density"] = float((np.hypot(gx, gy) > 0.1).mean())
        for k, v in enumerate(_hog(gray, cells=hog_cells)):
            row[f"hog_{k}"] = float(v)
        rows.append(row)
    if skipped:
        import warnings

        from ..core import AutomatedCleaningWarning
        warnings.warn(f"Skipped {len(skipped)} unreadable image(s): {skipped[:5]}", AutomatedCleaningWarning,
                      stacklevel=2)
    return pl.DataFrame(rows)


def augment(image, *, seed: int = 0, flip: bool = True, max_rotate: float = 15.0, crop: float = 0.1,
            brightness: float = 0.2, contrast: float = 0.2):
    """One reproducible random augmentation of a PIL image (same seed -> same result)."""
    Image, ImageOps = _pil()
    from PIL import ImageEnhance
    rng = np.random.default_rng(seed)
    im = image.copy()
    if flip and rng.random() < 0.5:
        im = ImageOps.mirror(im)
    if max_rotate:
        im = im.rotate(float(rng.uniform(-max_rotate, max_rotate)), resample=Image.Resampling.BILINEAR)
    if crop:
        w, h = im.size
        f = 1 - float(rng.uniform(0, crop))
        cw, ch = int(w * f), int(h * f)
        x, y = int(rng.integers(0, w - cw + 1)), int(rng.integers(0, h - ch + 1))
        im = im.crop((x, y, x + cw, y + ch)).resize((w, h), Image.Resampling.BILINEAR)
    if brightness:
        im = ImageEnhance.Brightness(im).enhance(float(rng.uniform(1 - brightness, 1 + brightness)))
    if contrast:
        im = ImageEnhance.Contrast(im).enhance(float(rng.uniform(1 - contrast, 1 + contrast)))
    return im
