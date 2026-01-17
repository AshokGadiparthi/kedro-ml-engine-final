"""Pipeline registry for Kedro 1.1.1."""

"""Pipeline registry with Phase 1 & 2 pipelines."""

from kedro.pipeline import Pipeline, pipeline

from ml_engine.pipelines import (
    create_data_loading_pipeline,
    create_data_validation_pipeline,
    create_data_cleaning_pipeline,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register all pipelines."""

    # Phase 1: Data Loading, Validation, Cleaning
    data_loading = create_data_loading_pipeline()
    data_validation = create_data_validation_pipeline()
    data_cleaning = create_data_cleaning_pipeline()

    # Phase 2: Feature Engineering
    from ml_engine.pipelines.feature_engineering import (
        handle_missing_values_node,
        scale_features_node,
        create_polynomial_features_node,
        create_interaction_features_node,
        generate_feature_statistics_node,
    )

    feature_engineering = pipeline(
        [
            handle_missing_values_node,
            scale_features_node,
            create_polynomial_features_node,
            create_interaction_features_node,
            generate_feature_statistics_node,
        ]
    )

    # Phase 2: Feature Selection
    from ml_engine.pipelines.feature_selection import (
        calculate_correlations_node,
        select_features_by_correlation_node,
        calculate_feature_importance_node,
        select_top_features_node,
    )

    feature_selection = pipeline(
        [
            calculate_correlations_node,
            select_features_by_correlation_node,
            calculate_feature_importance_node,
            select_top_features_node,
        ]
    )

    # Default pipeline: All phases
    default = (
            data_loading
            + data_validation
            + data_cleaning
            + feature_engineering
            + feature_selection
    )

    return {
        "default": default,
        "data_loading": data_loading,
        "data_validation": data_validation,
        "data_cleaning": data_cleaning,
        "feature_engineering": feature_engineering,
        "feature_selection": feature_selection,
        "__default__": default,
    }
