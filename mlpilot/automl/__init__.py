"""AutoML: module 5 of the platform.

* ``AutoML(target)``   classification / regression (incl. time series): task detection,
                       leakage check, cleaning + feature engineering, model search with
                       cross-validation and a time budget, ensembling, evaluation, save/load.
* ``AutoCluster()``    unlabelled data: choose the clustering method and number of clusters.
* ``automl(df, target=None)``  one call that picks the right one.
* ``detect_task``      what kind of problem a dataset is.
"""

from .automl import DEFAULT_METRICS, AutoML
from .cluster import AutoCluster
from .models import available_models
from .tasks import TaskInfo, detect_task


def load_model(path, *, public_key=None):
    """Load any saved AutoML / AutoCluster model directory (the checksum is verified first).

    With ``public_key`` (or the ``AC_MODEL_PUBLIC_KEY`` environment variable) the model's
    Ed25519 signature is verified before anything is unpickled (see ``security.sign_model``)."""
    import json
    from pathlib import Path

    from ..security import verify_if_required
    verify_if_required(path, public_key)
    manifest = json.loads((Path(path) / "manifest.json").read_text(encoding="utf-8"))
    kind = manifest.get("kind")
    if kind == "supervised":
        return AutoML.load(path)
    if kind == "clustering":
        return AutoCluster.load(path)
    if kind == "forecast":
        from ..forecast import Forecaster
        return Forecaster.load(path)
    raise ValueError(f"{path} is not a saved mlpilot model")


def automl(df, target: str | None = None, **kwargs):
    """Fit ``AutoML(target)`` when a target is given, otherwise ``AutoCluster()``."""
    if target is None:
        return AutoCluster(**kwargs).fit(df)
    return AutoML(target, **kwargs).fit(df)


__all__ = ["AutoML", "AutoCluster", "automl", "load_model", "detect_task", "TaskInfo", "available_models",
           "DEFAULT_METRICS"]
