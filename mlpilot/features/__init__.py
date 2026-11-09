"""Feature engineering: module 4 of the platform, built on the shared pipeline core.

Every class is a pipeline ``Step`` (fit on training data, replay on new data, saved as JSON):

* ``DateTimeFeatures``   calendar parts, cyclical encodings, elapsed days
* ``TargetEncode``       leakage-safe (out-of-fold) target encoding
* ``Interactions``       products / ratios of the most informative numeric columns
* ``GroupAggregations``  per-group statistics (mean, std, count, ...)
* ``LagFeatures``        time-series lags and rolling means
* ``TextEmbeddings``     TF-IDF + SVD (built in) or sentence-transformers vectors
* ``SelectFeatures``     importance / mutual-information / variance selection
* ``DropLeakage``        detect and drop columns that leak the target
  (``detect_leakage(df, target)`` runs the check on its own)

``default_feature_pipeline(target)`` wires them together. One-hot / ordinal / frequency
encoding and rare-category grouping come from the cleaning module (``EncodeCategorical``,
``GroupRareCategories``).
"""

from .dates import DATE_PARTS, DateTimeFeatures
from .defaults import DEFAULT_FEATURE_STEPS, default_feature_pipeline
from .encoding import TargetEncode
from .generators import GroupAggregations, Interactions, LagFeatures
from .leakage import DropLeakage, LeakageError, LeakageFinding, LeakageReport, detect_leakage
from .selection import SelectFeatures
from .text import TextEmbeddings

__all__ = [
    "DateTimeFeatures", "DATE_PARTS", "TargetEncode", "Interactions", "GroupAggregations",
    "LagFeatures", "TextEmbeddings", "SelectFeatures", "DropLeakage", "detect_leakage",
    "LeakageReport", "LeakageFinding", "LeakageError", "default_feature_pipeline",
    "DEFAULT_FEATURE_STEPS",
]
