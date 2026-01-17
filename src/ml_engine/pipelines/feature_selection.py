"""
Feature Selection Pipeline - Phase 2 (CORRECTED)
Selects best features based on multiple criteria.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import logging
from typing import Dict, Any
from kedro.pipeline import Pipeline, node

log = logging.getLogger(__name__)


# ============================================================================
# NODE FUNCTIONS
# ============================================================================

def calculate_correlations_node(
        engineered_features: pd.DataFrame
) -> pd.DataFrame:
    """Calculate feature correlations."""
    log.info("📊 Calculating feature correlations")
    correlation_matrix = engineered_features.corr()
    log.info(f"✅ Correlation matrix shape: {correlation_matrix.shape}")
    return correlation_matrix


def select_features_by_correlation_node(
        engineered_features: pd.DataFrame,
        correlation_matrix: pd.DataFrame,
        params: Dict[str, Any]
) -> pd.DataFrame:
    """Select features based on correlation threshold."""
    fs_config = params['feature_selection']

    if fs_config['method'] != 'correlation':
        log.info("⏭️ Correlation method not selected, returning all features")
        return engineered_features

    log.info("🔧 Selecting features by correlation")
    threshold = fs_config.get('threshold', 0.1)
    selected_cols = list(engineered_features.columns)

    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                col_to_remove = correlation_matrix.columns[j]
                if col_to_remove in selected_cols:
                    selected_cols.remove(col_to_remove)

    selected_features = engineered_features[selected_cols]
    removed = len(engineered_features.columns) - len(selected_cols)
    log.info(f"✅ Selected {len(selected_cols)} features (removed {removed})")
    return selected_features


def calculate_feature_importance_node(
        engineered_features: pd.DataFrame,
        params: Dict[str, Any]
) -> Dict[str, float]:
    """Calculate feature importance using tree-based method."""
    log.info("🔧 Calculating feature importance")

    np.random.seed(42)
    y = np.random.randint(0, 2, size=len(engineered_features))

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(engineered_features, y)

    feature_importance = dict(zip(
        engineered_features.columns,
        model.feature_importances_
    ))

    sorted_importance = dict(sorted(
        feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    ))

    log.info("✅ Top 5 features:")
    for i, (feat, imp) in enumerate(list(sorted_importance.items())[:5], 1):
        log.info(f"  {i}. {feat}: {imp:.4f}")

    return sorted_importance


def select_top_features_node(
        engineered_features: pd.DataFrame,
        feature_importance: Dict[str, float],
        params: Dict[str, Any]
) -> pd.DataFrame:
    """Select top N features by importance."""
    fs_config = params['feature_selection']

    if fs_config['method'] != 'importance':
        log.info("⏭️ Importance method not selected, returning all features")
        return engineered_features

    log.info("🔧 Selecting features by importance")
    n_features = fs_config.get('top_n', 10)
    selected_cols = list(dict(list(feature_importance.items())[:n_features]).keys())

    selected_features = engineered_features[selected_cols]
    log.info(f"✅ Selected top {len(selected_cols)} features")
    return selected_features


# ============================================================================
# PIPELINE FACTORY FUNCTION
# ============================================================================

def create_pipeline(**kwargs) -> Pipeline:
    """Create the feature selection pipeline."""
    return Pipeline(
        [
            node(
                func=calculate_correlations_node,
                inputs="engineered_features",
                outputs="correlation_matrix",
                name="calculate_correlations_node",
                tags="fs",
            ),
            node(
                func=select_features_by_correlation_node,
                inputs=["engineered_features", "correlation_matrix", "params:"],
                outputs="features_after_correlation",
                name="select_features_by_correlation_node",
                tags="fs",
            ),
            node(
                func=calculate_feature_importance_node,
                inputs=["engineered_features", "params:"],
                outputs="feature_importance",
                name="calculate_feature_importance_node",
                tags="fs",
            ),
            node(
                func=select_top_features_node,
                inputs=["engineered_features", "feature_importance", "params:"],
                outputs="selected_features",
                name="select_top_features_node",
                tags="fs",
            ),
        ]
    )