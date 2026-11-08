"""Pipeline steps that enforce data quality inside a ``Pipeline``.

* ``ValidateContract``: check every batch against a data contract. With no contract given,
  one is inferred from the training data during ``fit`` and saved inside the pipeline.
* ``QualityGate``: stop (or warn) when the quality score or high-severity issues cross a limit.

Both pass data through unchanged; put them first to guard raw input, last to guard output.
"""

from __future__ import annotations

import warnings

from ..core import AutomatedCleaningWarning, Step, register_step
from .checks import DIMENSIONS, check_quality
from .contracts import Contract, ContractViolation


class QualityWarning(AutomatedCleaningWarning):
    """Warning category for contract or quality-gate failures with ``on_fail="warn"``."""


class QualityGateError(Exception):
    """Data didn't meet a ``QualityGate``. ``.report`` holds the QualityReport."""

    def __init__(self, message: str, report):
        self.report = report
        super().__init__(message)


def _warn(message: str):
    with warnings.catch_warnings():   # show even if the host silenced warnings
        warnings.simplefilter("always", QualityWarning)
        warnings.warn(message, QualityWarning, stacklevel=5)


@register_step
class ValidateContract(Step):
    """Validate each batch against a data contract (inferred from training data if not given)."""

    name = "validate_contract"

    def __init__(self, *, contract=None, on_fail: str = "raise", infer_options: dict | None = None,
                 fit_only=None):
        if on_fail not in ("raise", "warn"):
            raise ValueError("on_fail must be 'raise' or 'warn'")
        if contract is not None and not isinstance(contract, (dict, str)):
            if isinstance(contract, Contract):
                contract = contract.to_dict()
            else:
                contract = str(contract)          # a path
        super().__init__(fit_only=fit_only, contract=contract, on_fail=on_fail,
                         infer_options=infer_options)
        self.last_result_ = None

    def _fit(self, df):
        given = self.params["contract"]
        if given is None:
            contract = Contract.infer(df, **(self.params["infer_options"] or {}))
        elif isinstance(given, dict):
            contract = Contract.from_dict(given)
        else:
            contract = Contract.from_file(given)
        # Store the contract itself so a saved pipeline doesn't need the original file.
        return {"contract": contract.to_dict()}

    @property
    def contract(self) -> Contract:
        return Contract.from_dict(self.state_["contract"])

    def _transform(self, df):
        result = self.contract.validate(df)
        self.last_result_ = result
        if not result.passed:
            if self.params["on_fail"] == "raise":
                raise ContractViolation(result)
            _warn(result.summary())
        return df


@register_step
class QualityGate(Step):
    """Stop (or warn) when the quality score, a dimension or high-severity issues cross a limit."""

    name = "quality_gate"

    def __init__(self, *, min_score: float | None = 80.0, max_high_issues: int | None = None,
                 min_dimensions: dict | None = None, on_fail: str = "raise", fit_only=None):
        if on_fail not in ("raise", "warn"):
            raise ValueError("on_fail must be 'raise' or 'warn'")
        unknown = set(min_dimensions or {}) - set(DIMENSIONS)
        if unknown:
            raise ValueError(f"Unknown dimension(s) {sorted(unknown)}; choose from {DIMENSIONS}")
        super().__init__(fit_only=fit_only, min_score=min_score, max_high_issues=max_high_issues,
                         min_dimensions=min_dimensions, on_fail=on_fail)
        self.last_report_ = None

    def _transform(self, df):
        report = check_quality(df)
        self.last_report_ = report
        p = self.params
        problems = []
        if p["min_score"] is not None and report.score < p["min_score"]:
            problems.append(f"score {report.score:.1f} < {p['min_score']}")
        if p["max_high_issues"] is not None and report.counts["high"] > p["max_high_issues"]:
            problems.append(f"{report.counts['high']} high-severity issues > {p['max_high_issues']}")
        for dim, minimum in (p["min_dimensions"] or {}).items():
            if report.dimensions[dim] < minimum:
                problems.append(f"{dim} {report.dimensions[dim]:.1f} < {minimum}")
        if problems:
            message = "Quality gate failed: " + "; ".join(problems)
            if p["on_fail"] == "raise":
                raise QualityGateError(message, report)
            _warn(message)
        return df
