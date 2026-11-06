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


def automl(df, target: str | None = None, **kwargs):
    """Fit ``AutoML(target)`` when a target is given, otherwise ``AutoCluster()``."""
    if target is None:
        return AutoCluster(**kwargs).fit(df)
    return AutoML(target, **kwargs).fit(df)


__all__ = ["AutoML", "AutoCluster", "automl", "detect_task", "TaskInfo", "available_models",
           "DEFAULT_METRICS"]
