"""Pipeline registry with Phase 1 & 2 pipelines."""

from kedro.pipeline import Pipeline, pipeline, node

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

    # Phase 2: Feature Engineering
    from ml_engine.pipelines.feature_engineering import (
        handle_missing_values_node,
        scale_features_node,
        create_polynomial_features_node,
        create_interaction_features_node,
        generate_feature_statistics_node,
    )

    feature_engineering_pipeline = pipeline(
        [
            handle_missing_values_node,
            scale_features_node,
            create_polynomial_features_node,
            create_interaction_features_node,
            generate_feature_statistics_node,
        ],
        tags="fe",
    )

    # Phase 2: Feature Selection
    from ml_engine.pipelines.feature_selection import (
        calculate_correlations_node,
        select_features_by_correlation_node,
        calculate_feature_importance_node,
        select_top_features_node,
    )

    feature_selection_pipeline = pipeline(
        [
            calculate_correlations_node,
            select_features_by_correlation_node,
            calculate_feature_importance_node,
            select_top_features_node,
        ],
        tags="fs",
    )

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