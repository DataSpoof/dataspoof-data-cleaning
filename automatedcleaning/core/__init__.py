"""Shared pipeline core for the DataSpoof data-science modules.

This subpackage deliberately imports nothing from the rest of ``automatedcleaning``: cleaning
is just the first module built on it, and the core can be lifted into its own package when
feature engineering, AutoML, etc. are added.

Concepts
--------
``Step``      learns rules in ``fit`` (JSON-serialisable ``state_``) and replays them in
              ``transform``.
``Pipeline``  ordered steps; ``fit_transform`` on training data, ``transform`` on anything
              later; ``save``/``load`` as JSON; a ``RunReport`` per run.
Registry      config files refer to steps by name (``register_step``, ``register_function``).
"""

from .pipeline import (FORMAT, FORMAT_VERSION, AutomatedCleaningWarning, Pipeline, PipelineStepError,
                       PipelineWarning, ServiceWarning,
                       RunReport, SchemaMismatchError, StepReport, make_step)
from .registry import (available_steps, get_step_class, register_function,
                       register_function_namespace, register_step, resolve_function)
from .step import FunctionStep, NotFittedError, Step

__all__ = [
    "Step", "FunctionStep", "Pipeline", "RunReport", "StepReport", "make_step",
    "register_step", "register_function", "register_function_namespace", "available_steps",
    "get_step_class", "resolve_function", "NotFittedError", "SchemaMismatchError",
    "PipelineWarning", "PipelineStepError", "AutomatedCleaningWarning", "ServiceWarning", "FORMAT", "FORMAT_VERSION",
]
