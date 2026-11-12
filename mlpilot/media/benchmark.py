"""Robustness benchmark for image models: weather, low light, blur, noise and covered faces.

How well does a model hold up in fog, rain, snow, at night, through a dirty lens, or when faces
are partly covered? Re-score it on degraded copies of a test set, one condition and severity
at a time.

>>> from mlpilot.media import benchmark
>>> predict = benchmark.image_classifier(model)                 # a fitted AutoML on image_features
>>> rep = benchmark.condition_benchmark(predict, test_images, test_labels)
>>> print(rep.summary())                                       # clean score, then the worst conditions
>>> rep.table                                                  # condition x severity -> score, drop
>>> rep.passed(max_drop=0.10)                                  # gate a release on robustness
>>> rep = benchmark.condition_benchmark(benchmark.plate_reader("IN"), cars, plate_texts)

Works with anything that maps a list of PIL images to a list of predictions: an AutoML model on
``image_features``, a plate reader, a face detector (``labels`` = expected face counts), or your
own deep model. ``score`` defaults to exact-match accuracy; pass any ``score(y_true, y_pred)``.

``condition_augment`` produces the same degraded copies for *training*, so the model sees the
conditions it is weakest at.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np
import polars as pl

from .conditions import SIMULATED, _as_pil, apply_condition
from .image import _items

FACE_OCCLUDERS = ("face_hair", "face_mask", "face_sunglasses", "face_cutout")


def _accuracy(y_true, y_pred) -> float:
    return float(np.mean([str(a) == str(b) for a, b in zip(y_true, y_pred)])) if len(y_true) else float("nan")


def _degrade(image, condition: str, severity: float, seed: int):
    if condition.startswith("face_"):
        from .faces import add_occluder
        return add_occluder(image, condition[5:], severity=severity, seed=seed)
    return apply_condition(image, condition, severity=severity, seed=seed)


def _load(images) -> list:
    if isinstance(images, (list, tuple)) and images and not isinstance(images[0], str):
        return [_as_pil(im) for im in images]
    return [_as_pil(path if im is None else im) for path, im in _items(images)]


@dataclasses.dataclass
class BenchmarkReport:
    clean: float                     # score on the unmodified images
    table: pl.DataFrame              # condition, severity, score, drop (clean - score), n
    by_condition: pl.DataFrame       # condition, mean_score, worst_score, worst_drop
    errors: dict                     # {condition: error message} for conditions that could not run

    @property
    def weakest(self) -> list:
        """Conditions ordered from the largest worst-case drop down."""
        return self.by_condition.sort("worst_drop", descending=True)["condition"].to_list()

    def passed(self, *, max_drop: float = 0.10) -> bool:
        return bool(self.by_condition.is_empty() or self.by_condition["worst_drop"].max() <= max_drop)

    def summary(self, top: int = 5) -> str:
        lines = [f"Clean score: {self.clean:.3f}"]
        for r in self.by_condition.sort("worst_drop", descending=True).head(top).iter_rows(named=True):
            lines.append(f"  {r['condition']:<16} worst {r['worst_score']:.3f} (drop {r['worst_drop']:+.3f}), "
                         f"mean {r['mean_score']:.3f}")
        for c, e in self.errors.items():
            lines.append(f"  {c:<16} skipped: {e}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        weak = ", ".join(self.weakest[:3])
        return f"BenchmarkReport(clean={self.clean:.3f}, conditions={self.by_condition.height}, weakest=[{weak}])"


def condition_benchmark(predict: Callable, images, labels, *, conditions=None, severities=(1, 3, 5),
                        score: Callable | None = None, faces: bool = False, seed: int = 0) -> BenchmarkReport:
    """Score ``predict`` on clean and degraded copies of ``images``.

    ``predict(list_of_pil_images) -> predictions``; ``labels`` are the expected predictions.
    ``conditions`` default to every simulated condition (fog, rain, snow, low_light, glare,
    motion_blur, noise, jpeg, shadow, cutout); ``faces=True`` adds hair / mask / sunglasses /
    cutout coverings over detected faces (needs ``[vision]``)."""
    score = score or _accuracy
    imgs = _load(images)
    labels = list(labels)
    if len(imgs) != len(labels):
        raise ValueError(f"{len(imgs)} images but {len(labels)} labels")
    conds = list(conditions) if conditions is not None else list(SIMULATED)
    if faces:
        conds += [c for c in FACE_OCCLUDERS if c not in conds]
    clean = float(score(labels, list(predict(imgs))))
    rows, errors = [], {}
    for c in conds:
        try:
            for s in severities:
                degraded = [_degrade(im, c, s, seed + i) for i, im in enumerate(imgs)]
                val = float(score(labels, list(predict(degraded))))
                rows.append({"condition": c, "severity": float(s), "score": val, "drop": clean - val, "n": len(imgs)})
        except Exception as exc:  # noqa: BLE001 - e.g. face coverings without OpenCV; report and go on
            errors[c] = f"{type(exc).__name__}: {exc}"
            rows = [r for r in rows if r["condition"] != c]
    schema = {"condition": pl.Utf8, "severity": pl.Float64, "score": pl.Float64, "drop": pl.Float64, "n": pl.Int64}
    table = pl.DataFrame(rows, schema=schema)
    by = (table.group_by("condition", maintain_order=True)
          .agg(pl.col("score").mean().alias("mean_score"), pl.col("score").min().alias("worst_score"),
               pl.col("drop").max().alias("worst_drop")))
    return BenchmarkReport(clean=clean, table=table, by_condition=by, errors=errors)


def condition_augment(images, labels=None, *, conditions=None, severities=(2, 4), copies: int = 1,
                      seed: int = 0) -> tuple:
    """Training augmentation: ``copies`` degraded versions of every image, each with a random
    condition and severity. Returns ``(images, labels, conditions_used)`` *including* the originals,
    so ``labels`` stay aligned (``None`` labels are passed through as ``None``)."""
    rng = np.random.default_rng(seed)
    imgs = _load(images)
    labels = list(labels) if labels is not None else [None] * len(imgs)
    conds = list(conditions) if conditions is not None else list(SIMULATED)
    out_i, out_l, used = list(imgs), list(labels), ["clean"] * len(imgs)
    for k in range(copies):
        for i, (im, y) in enumerate(zip(imgs, labels)):
            c = conds[int(rng.integers(len(conds)))]
            s = float(rng.uniform(min(severities), max(severities)))
            out_i.append(_degrade(im, c, s, seed * 7919 + k * len(imgs) + i))
            out_l.append(y)
            used.append(c)
    return out_i, out_l, used


# ---- ready-made predictors -------------------------------------------------------------------
def image_classifier(model, *, features: Callable | None = None, keep_path: bool = False,
                     **feature_kwargs) -> Callable:
    """Wrap a model trained on :func:`mlpilot.media.image_features` (an ``AutoML`` or anything with
    ``predict(DataFrame)``) as ``predict(images)``. Pass the same ``feature_kwargs`` used in training;
    the ``path`` column is dropped unless ``keep_path=True``."""
    from .image import image_features
    feats = features or image_features

    def predict(imgs: list) -> list:
        df = feats(list(imgs), **feature_kwargs)
        if not keep_path and "path" in df.columns:
            df = df.drop("path")
        out = model.predict(df)
        return list(out.to_list() if hasattr(out, "to_list") else out)
    return predict


def plate_reader(pattern=None, *, enhance: bool = False) -> Callable:
    """``predict(images)`` -> the text of the most confident plate in each image ("" if none)."""
    from .plates import detect_plates

    def predict(imgs: list) -> list:
        out = []
        for im in imgs:
            found = detect_plates(im, pattern=pattern, enhance=enhance)
            out.append(found[0].text if found else "")
        return out
    return predict


def face_counter(**kwargs) -> Callable:
    """``predict(images)`` -> number of faces found in each image (labels = expected counts)."""
    from .faces import detect_faces
    return lambda imgs: [len(detect_faces(im, eyes=False, **kwargs)) for im in imgs]
