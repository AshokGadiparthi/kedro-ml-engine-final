"""Pipeline registry with Phase 1 & 2 pipelines (CORRECTED)."""

from kedro.pipeline import Pipeline

from ml_engine.pipelines import (
    create_data_loading_pipeline,
    create_data_validation_pipeline,
    create_data_cleaning_pipeline,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register all pipelines."""

    # Phase 1: Data Loading, Validation, Cleaning
    data_loading_pipeline = create_data_loading_pipeline()
    data_validation_pipeline = create_data_validation_pipeline()
    data_cleaning_pipeline = create_data_cleaning_pipeline()

    # Phase 2: Feature Engineering (NOW USING create_pipeline FUNCTION)
    from ml_engine.pipelines.feature_engineering import create_pipeline as create_fe_pipeline
    feature_engineering_pipeline = create_fe_pipeline()

    # Phase 2: Feature Selection (NOW USING create_pipeline FUNCTION)
    from ml_engine.pipelines.feature_selection import create_pipeline as create_fs_pipeline
    feature_selection_pipeline = create_fs_pipeline()

    # Combine all pipelines
    default_pipeline = (
            data_loading_pipeline
            + data_validation_pipeline
            + data_cleaning_pipeline
            + feature_engineering_pipeline
            + feature_selection_pipeline
    )

    return {
        "__default__": default_pipeline,
        "default": default_pipeline,
        "data_loading": data_loading_pipeline,
        "data_validation": data_validation_pipeline,
        "data_cleaning": data_cleaning_pipeline,
        "feature_engineering": feature_engineering_pipeline,
        "feature_selection": feature_selection_pipeline,
        "data_processing": data_loading_pipeline + data_validation_pipeline + data_cleaning_pipeline,
        "feature_processing": feature_engineering_pipeline + feature_selection_pipeline,
    }