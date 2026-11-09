"""Evaluation & explainability: metrics, SHAP explanations, error analysis, fairness, model cards.

Every function takes a fitted ``AutoML`` model (or a scikit-learn estimator plus
``target=...``) and labelled data::

    report = ac.evaluate_model(model, test)            # metrics, confusion matrix, calibration
    shap = ac.explain(model, test)                     # per-row + global SHAP values
    errors = ac.error_analysis(model, test)            # worst predictions & failing segments
    fair = ac.fairness_report(model, test, ["Sex"])    # bias across groups
    ac.model_card(model, test, sensitive=["Sex"]).save("card.html")
"""

from .adapter import ModelAdapter
from .card import ModelCard, model_card
from .errors import ErrorReport, error_analysis
from .explain import Explanation, explain, partial_dependence, shapley_values
from .fairness import DEFAULT_LIMITS, FairnessReport, FairnessResult, cramers_v, fairness_report, find_proxies
from .metrics import EvaluationReport, evaluate_model

__all__ = ["ModelAdapter", "ModelCard", "model_card", "ErrorReport", "error_analysis", "Explanation",
           "explain", "partial_dependence", "shapley_values", "DEFAULT_LIMITS", "FairnessReport",
           "FairnessResult", "cramers_v", "fairness_report", "find_proxies", "EvaluationReport",
           "evaluate_model"]
