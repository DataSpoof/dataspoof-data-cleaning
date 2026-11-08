"""Experiment tracking & model registry (module 7).

* ``Tracker``        runs with params, metrics (with steps), datasets (fingerprints), artifacts,
                     git commit and user; ``run.log_automl(model, train, test)`` logs it all.
* ``ModelRegistry``  versioned models promoted dev → staging → production with metric gates,
                     an audit history and rollback.

Both are plain directories of JSON files — no server, easy to back up or put on a shared drive.
"""

from ._store import dataset_fingerprint
from .models import STAGES, ModelRegistry, ModelVersion, PromotionError
from .tracking import Run, Tracker

__all__ = ["Tracker", "Run", "ModelRegistry", "ModelVersion", "PromotionError", "STAGES", "dataset_fingerprint"]
