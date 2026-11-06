from .cleaning import *
from .steps import *
from . import cleaning as _cleaning, steps as _steps
from .core import (FunctionStep, NotFittedError, Pipeline, PipelineStepError, PipelineWarning,
                   RunReport, SchemaMismatchError, Step, available_steps, register_function,
                   register_function_namespace, register_step)
from .cleaners import *
from . import quality
from .quality import (ColumnProfile, Contract, ContractError, ContractResult, ContractViolation,
                      Issue, Profile, QualityGate, QualityGateError, QualityReport, QualityWarning,
                      SchemaDiff, ValidateContract, check_quality, compare_schemas, infer_schema,
                      profile)
from . import features
from .features import (DateTimeFeatures, DropLeakage, GroupAggregations, Interactions, LagFeatures,
                       LeakageError, LeakageReport, SelectFeatures, TargetEncode, TextEmbeddings,
                       default_feature_pipeline, detect_leakage)
from . import automl as automl_module
from .automl import AutoCluster, AutoML, TaskInfo, automl, available_models, detect_task
from . import evaluation
from .evaluation import (ErrorReport, EvaluationReport, Explanation, FairnessReport, ModelCard,
                         error_analysis, evaluate_model, explain, fairness_report, model_card,
                         partial_dependence)

# Every function in steps.py works as a pipeline step by name, e.g. {"step": "normalize_unicode"}.
register_function_namespace(_steps)
# Legacy cleaning.py functions too, except those that prompt for input, write files,
# or don't return a cleaned DataFrame.
register_function_namespace(_cleaning, exclude={
    "clean_data", "load_data", "save_cleaned_data", "save_dashboard", "generate_dashboard",
    "fix_spelling_errors_in_columns", "fix_spelling_errors_in_categorical",
    "correct_categorical_values", "manual_correction", "check_and_handle_imbalance",
    "check_data_types", "check_outliers", "check_skewness", "check_problem_type",
    "print_header", "print_section_header", "preprocess_text", "fix_incorrect_data_types",
})
