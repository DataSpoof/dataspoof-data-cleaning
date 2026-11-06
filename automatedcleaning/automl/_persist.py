"""Saving and loading trained models.

A saved model is a directory:

* ``manifest.json``   what was trained, how it scored, library versions, and the SHA-256 of
                      the model file;
* ``model.joblib``    the fitted scikit-learn estimator (trained models can't be expressed as
                      JSON, so joblib/pickle is the standard format);
* ``preprocess.json`` the fitted cleaning/feature pipeline (plain JSON, as everywhere else).

Loading verifies the checksum before unpickling, which catches a corrupted or swapped model
file. Unpickling still runs code, so only load models produced by people you trust.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import warnings
from pathlib import Path

import polars as pl

from ..core import Pipeline
from ..quality.checks import jsonable

FORMAT = "dataspoof.automl"
FORMAT_VERSION = 1
MODEL_FILE, MANIFEST_FILE, PREPROCESS_FILE = "model.joblib", "manifest.json", "preprocess.json"


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def versions() -> dict:
    from importlib import metadata
    out = {"python": platform.python_version(), "polars": pl.__version__}
    for pkg in ("scikit-learn", "automatedcleaning", "lightgbm", "xgboost"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


def save_artifact(path, kind: str, model, manifest: dict, preprocess: Pipeline | None) -> Path:
    import joblib

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path / MODEL_FILE)
    if preprocess is not None:
        preprocess.save(path / PREPROCESS_FILE)
    full = {"format": FORMAT, "format_version": FORMAT_VERSION, "kind": kind, **manifest,
            "saved_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "versions": versions(), "model_file": MODEL_FILE,
            "model_sha256": file_sha256(path / MODEL_FILE)}
    (path / MANIFEST_FILE).write_text(json.dumps(jsonable(full), indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    return path


def load_artifact(path, kind: str):
    """Return ``(manifest, model, preprocess)`` after verifying the model checksum."""
    import joblib

    path = Path(path)
    manifest = json.loads((path / MANIFEST_FILE).read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT or manifest.get("kind") != kind:
        raise ValueError(f"{path} is not a saved {kind} model")
    if manifest.get("format_version", 0) > FORMAT_VERSION:
        raise ValueError(f"{path} was saved by a newer version; upgrade automatedcleaning")
    model_path = path / manifest["model_file"]
    if file_sha256(model_path) != manifest["model_sha256"]:
        raise ValueError(f"{model_path} doesn't match the checksum in its manifest; it may be "
                         f"corrupted or replaced. Refusing to load it.")
    saved = manifest.get("versions", {}).get("scikit-learn")
    current = versions().get("scikit-learn")
    if saved and current and saved != current:
        warnings.warn(f"Model was trained with scikit-learn {saved}; you have {current}. "
                      f"Predictions may differ or loading may fail.", UserWarning, stacklevel=3)
    model = joblib.load(model_path)
    preprocess = Pipeline.load(path / PREPROCESS_FILE) if (path / PREPROCESS_FILE).exists() else None
    return manifest, model, preprocess
