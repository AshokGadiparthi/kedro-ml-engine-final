"""
Feature Selection Pipeline - Phase 2
Selects best features based on multiple criteria.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
import logging
from typing import Dict, Any

log = logging.getLogger(__name__)


def calculate_correlations_node(
        engineered_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate feature correlations.

    Args:
        engineered_data: Input DataFrame

    Returns:
        correlation_matrix: Correlation matrix
    """
    log.info("📊 Calculating feature correlations")

    correlation_matrix = engineered_data.corr()

    log.info(f"✅ Correlation matrix shape: {correlation_matrix.shape}")
    return correlation_matrix


def select_features_by_correlation_node(
        engineered_data: pd.DataFrame,
        correlation_matrix: pd.DataFrame,
        params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Select features based on correlation threshold.

    Args:
        engineered_data: Input DataFrame
        correlation_matrix: Correlation matrix
        params: Pipeline parameters

    Returns:
        selected_features: DataFrame with selected features
    """
    fs_config = params['feature_selection']

    if fs_config['method'] != 'correlation':
        log.info("⏭️ Correlation method not selected, returning all features")
        return engineered_data

    log.info("🔧 Selecting features by correlation")

    threshold = fs_config.get('threshold', 0.1)
    selected_cols = list(engineered_data.columns)

    # Remove highly correlated features
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                col_to_remove = correlation_matrix.columns[j]
                if col_to_remove in selected_cols:
                    selected_cols.remove(col_to_remove)
                    log.info(f"Removing {col_to_remove} (corr: {correlation_matrix.iloc[i, j]:.3f})")

    selected_features = engineered_data[selected_cols]

    removed = len(engineered_data.columns) - len(selected_cols)
    log.info(f"✅ Selected {len(selected_cols)} features (removed {removed})")

    return selected_features


def calculate_feature_importance_node(
        engineered_data: pd.DataFrame,
        params: Dict[str, Any]
) -> Dict[str, float]:
    """
    Calculate feature importance using tree-based method.

    Args:
        engineered_data: Input DataFrame
        params: Pipeline parameters

    Returns:
        feature_importance: Feature importance scores
    """
    log.info("🔧 Calculating feature importance")

    # Create dummy target for demonstration
    np.random.seed(42)
    y = np.random.randint(0, 2, size=len(engineered_data))

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(engineered_data, y)

    feature_importance = dict(zip(
        engineered_data.columns,
        model.feature_importances_
    ))

    # Sort by importance
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
        engineered_data: pd.DataFrame,
        feature_importance: Dict[str, float],
        params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Select top N features by importance.

    Args:
        engineered_data: Input DataFrame
        feature_importance: Feature importance scores
        params: Pipeline parameters

    Returns:
        selected_features: DataFrame with top N features
    """
    fs_config = params['feature_selection']

    if fs_config['method'] != 'importance':
        log.info("⏭️ Importance method not selected, returning all features")
        return engineered_data

    log.info("🔧 Selecting features by importance")

    n_features = fs_config.get('top_n', 10)
    selected_cols = list(dict(list(feature_importance.items())[:n_features]).keys())

    selected_features = engineered_data[selected_cols]

    log.info(f"✅ Selected top {len(selected_cols)} features")
    return selected_features