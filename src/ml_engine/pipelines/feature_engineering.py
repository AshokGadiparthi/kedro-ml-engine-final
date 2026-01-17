"""
Feature Engineering Pipeline - Phase 2 (EXPERT CORRECTED)
Handles missing values, scaling, encoding, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import logging
from typing import Dict, Any
from kedro.pipeline import Pipeline, node

log = logging.getLogger(__name__)


# ============================================================================
# NODE FUNCTIONS
# ============================================================================

def handle_missing_values_node(
        cleaned_data: pd.DataFrame,
        params: Dict[str, Any]
) -> pd.DataFrame:
    """Handle missing values using configured strategy."""
    strategy_config = params['impute_strategy'] if isinstance(params.get('impute_strategy'), dict) else params
    method = strategy_config.get('method', 'mean') if isinstance(strategy_config, dict) else strategy_config

    log.info(f"🔧 Imputing missing values using: {method}")

    if cleaned_data.isnull().sum().sum() == 0:
        log.info("✅ No missing values found")
        return cleaned_data

    if method == 'mean':
        imputer = SimpleImputer(strategy='mean')
    elif method == 'median':
        imputer = SimpleImputer(strategy='median')
    elif method == 'forward_fill':
        imputed = cleaned_data.fillna(method='ffill', limit=3)
        return imputed
    elif method == 'knn':
        imputer = KNNImputer(n_neighbors=5)
    else:
        return cleaned_data

    imputed_array = imputer.fit_transform(cleaned_data)
    imputed_data = pd.DataFrame(imputed_array, columns=cleaned_data.columns)
    log.info(f"✅ Shape: {imputed_data.shape}, Missing: {imputed_data.isnull().sum().sum()}")
    return imputed_data


def scale_features_node(
        imputed_data: pd.DataFrame,
        params: Dict[str, Any]
) -> pd.DataFrame:
    """Scale features using configured method."""
    scaling_config = params.get('scaling', 'standard')
    method = scaling_config if isinstance(scaling_config, str) else scaling_config.get('method', 'standard')

    log.info(f"🔧 Scaling features using: {method}")

    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler(feature_range=(0, 1))
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        return imputed_data

    scaled_array = scaler.fit_transform(imputed_data)
    scaled_data = pd.DataFrame(scaled_array, columns=imputed_data.columns)
    log.info(f"✅ Shape: {scaled_data.shape}")
    return scaled_data


def create_polynomial_features_node(
        scaled_data: pd.DataFrame,
        params: Dict[str, Any]
) -> pd.DataFrame:
    """Create polynomial features if configured."""
    engineered_data = scaled_data.copy()
    degree = params.get('poly_degree', 2)

    if degree <= 1:
        return engineered_data

    log.info("🔧 Creating polynomial features")
    cols = list(engineered_data.columns)

    for col in cols:
        for d in range(2, degree + 1):
            new_col = f"{col}_poly{d}"
            engineered_data[new_col] = engineered_data[col] ** d

    log.info(f"✅ Shape: {engineered_data.shape}")
    return engineered_data


def create_interaction_features_node(
        polynomial_data: pd.DataFrame,
        params: Dict[str, Any]
) -> pd.DataFrame:
    """Create interaction features if configured."""
    final_engineered = polynomial_data.copy()

    if not params.get('interaction_only', False):
        return final_engineered

    log.info("🔧 Creating interaction features")
    non_poly_cols = [col for col in polynomial_data.columns if 'poly' not in col]
    interaction_count = 0
    max_interactions = params.get('max_interactions', 10)

    for i in range(len(non_poly_cols)):
        for j in range(i + 1, len(non_poly_cols)):
            col1, col2 = non_poly_cols[i], non_poly_cols[j]
            new_col = f"{col1}_x_{col2}"
            final_engineered[new_col] = polynomial_data[col1] * polynomial_data[col2]
            interaction_count += 1
            if interaction_count >= max_interactions:
                break
        if interaction_count >= max_interactions:
            break

    log.info(f"✅ Added {interaction_count} interactions, shape: {final_engineered.shape}")
    return final_engineered


def generate_feature_statistics_node(
        engineered_data: pd.DataFrame
) -> Dict[str, Any]:
    """Generate feature statistics."""
    log.info("📊 Generating feature statistics")

    feature_stats = {
        'shape': list(engineered_data.shape),
        'columns': engineered_data.columns.tolist(),
        'n_features': engineered_data.shape[1],
        'n_samples': engineered_data.shape[0],
        'missing_values': engineered_data.isnull().sum().to_dict(),
    }

    log.info(f"✅ Generated {len(feature_stats)} sections")
    return feature_stats


# ============================================================================
# PIPELINE FACTORY FUNCTION
# ============================================================================

def create_pipeline(**kwargs) -> Pipeline:
    """Create the feature engineering pipeline."""
    return Pipeline(
        [
            node(
                func=handle_missing_values_node,
                inputs=["cleaned_data", "params:feature_engineering"],  # ✅ FIXED
                outputs="imputed_data",
                name="handle_missing_values_node",
                tags="fe",
            ),
            node(
                func=scale_features_node,
                inputs=["imputed_data", "params:feature_engineering"],  # ✅ FIXED
                outputs="scaled_data",
                name="scale_features_node",
                tags="fe",
            ),
            node(
                func=create_polynomial_features_node,
                inputs=["scaled_data", "params:feature_engineering"],  # ✅ FIXED
                outputs="polynomial_data",
                name="create_polynomial_features_node",
                tags="fe",
            ),
            node(
                func=create_interaction_features_node,
                inputs=["polynomial_data", "params:feature_engineering"],  # ✅ FIXED
                outputs="engineered_features",
                name="create_interaction_features_node",
                tags="fe",
            ),
            node(
                func=generate_feature_statistics_node,
                inputs="engineered_features",
                outputs="feature_statistics",
                name="generate_feature_statistics_node",
                tags="fe",
            ),
        ]
    )