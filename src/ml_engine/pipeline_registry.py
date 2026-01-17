"""
Project pipelines.
"""

from typing import Dict
from kedro.pipeline import Pipeline, pipeline

# Import PERFECT Phase 2 pipeline factory functions
from ml_engine.pipelines.data_loading import create_pipeline as create_data_loading_pipeline
from ml_engine.pipelines.data_validation import create_pipeline as create_data_validation_pipeline
from ml_engine.pipelines.data_cleaning import create_pipeline as create_data_cleaning_pipeline
from ml_engine.pipelines.feature_engineering import create_pipeline as create_feature_engineering_pipeline
from ml_engine.pipelines.feature_selection import create_pipeline as create_feature_selection_pipeline

# Optionally import Phase 3 pipelines if they exist
try:
    from ml_engine.pipelines.model_training import create_pipeline as create_model_training_pipeline
except ImportError:
    create_model_training_pipeline = None


def create_pipelines(**kwargs) -> Dict[str, Pipeline]:
    """
    Create the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.

    """

    # Phase 1: Data Loading
    data_loading_pipeline = create_data_loading_pipeline()

    # Phase 1: Data Validation
    data_validation_pipeline = create_data_validation_pipeline()

    # Phase 1: Data Cleaning
    data_cleaning_pipeline = create_data_cleaning_pipeline()

    # Phase 2: Feature Engineering (PERFECT - with all 5 gaps fixed)
    feature_engineering_pipeline = create_feature_engineering_pipeline()

    # Phase 2: Feature Selection (PERFECT - with all 5 gaps fixed)
    feature_selection_pipeline = create_feature_selection_pipeline()

    # Phase 3: Model Training (optional, if it exists)
    if create_model_training_pipeline:
        model_training_pipeline = create_model_training_pipeline()
    else:
        model_training_pipeline = None

    # Combine pipelines
    pipelines = {
        # Individual pipelines (for selective running)
        "data_loading": data_loading_pipeline,
        "data_validation": data_validation_pipeline,
        "data_cleaning": data_cleaning_pipeline,
        "feature_engineering": feature_engineering_pipeline,
        "feature_selection": feature_selection_pipeline,

        # Phase 1: Complete data prep
        "__default__": (
                data_loading_pipeline +
                data_validation_pipeline +
                data_cleaning_pipeline
        ),

        # Phase 1 + Phase 2: Complete preprocessing
        "phase1_2": (
                data_loading_pipeline +
                data_validation_pipeline +
                data_cleaning_pipeline +
                feature_engineering_pipeline +
                feature_selection_pipeline
        ),
    }

    # Add Phase 3 if it exists
    if model_training_pipeline:
        pipelines["model_training"] = model_training_pipeline
        pipelines["phase1_2_3"] = (
                data_loading_pipeline +
                data_validation_pipeline +
                data_cleaning_pipeline +
                feature_engineering_pipeline +
                feature_selection_pipeline +
                model_training_pipeline
        )

    return pipelines