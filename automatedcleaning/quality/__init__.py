"""Profiling & data quality: module 2 of the platform, built on the shared pipeline core.

* ``profile(df)``         automated EDA: per-column stats, semantic types, correlations,
                          plain-English insights and the quality report.
* ``check_quality(df)``   every data-quality check plus a 0–100 score and grade, split into
                          completeness / uniqueness / validity / consistency and per column.
* ``Contract``            declarative data contracts (YAML/JSON); ``validate`` any batch;
                          ``Contract.infer`` writes one from data you trust.
* ``compare_schemas``     added / removed / retyped columns between two datasets.
* ``ValidateContract`` / ``QualityGate``   pipeline steps that enforce all of the above.
"""

from .checks import (DEFAULT_WEIGHTS, DIMENSIONS, Issue, QualityReport, check_quality, grade)
from .contracts import (Contract, ContractError, ContractResult, ContractViolation, RuleResult)
from .pipeline_steps import QualityGate, QualityGateError, QualityWarning, ValidateContract
from .profile import ColumnProfile, Profile, profile, semantic_type
from .schema import SchemaDiff, compare_schemas, infer_schema

__all__ = [
    "profile", "Profile", "ColumnProfile", "semantic_type",
    "check_quality", "QualityReport", "Issue", "grade", "DIMENSIONS", "DEFAULT_WEIGHTS",
    "Contract", "ContractResult", "RuleResult", "ContractError", "ContractViolation",
    "infer_schema", "compare_schemas", "SchemaDiff",
    "ValidateContract", "QualityGate", "QualityGateError", "QualityWarning",
]
