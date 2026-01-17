"""
Feature Engineering Pipeline - Phase 2 (CORRECTED)
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
    strategy_config = params['missing_value_strategy']
    method = strategy_config['method']

    log.info(f"🔧 Imputing missing values using: {method}")

    if cleaned_data.isnull().sum().sum() == 0:
        log.info("✅ No missing values found")
        return cleaned_data

    if method == 'mean':
        imputer = SimpleImputer(strategy='mean')
    elif method == 'median':
        imputer = SimpleImputer(strategy='median')
    elif method == 'forward_fill':
        limit = strategy_config.get('forward_fill_limit', 3)
        imputed = cleaned_data.fillna(method='ffill', limit=limit)
        return imputed
    elif method == 'knn':
        n_neighbors = strategy_config.get('knn_neighbors', 5)
        imputer = KNNImputer(n_neighbors=n_neighbors)
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
    scaling_config = params['scaling']
    method = scaling_config['method']

    log.info(f"🔧 Scaling features using: {method}")

    if method == 'standard':
        scaler = StandardScaler(
            with_mean=scaling_config.get('with_mean', True),
            with_std=scaling_config.get('with_std', True)
        )
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
    fe_config = params['feature_engineering']

    if not fe_config.get('create_polynomial', False):
        return engineered_data

    log.info("🔧 Creating polynomial features")
    degree = fe_config.get('polynomial_degree', 2)
    cols = engineered_data.columns

    for col in cols:
        for d in range(2, degree + 1):
            new_col = f"{col}_poly{d}"
            engineered_data[new_col] = engineered_data[col] ** d

    log.info(f"✅ Shape: {engineered_data.shape}")
    return engineered_data


def create_interaction_features_node(
        engineered_data: pd.DataFrame,
        params: Dict[str, Any]
) -> pd.DataFrame:
    """Create interaction features if configured."""
    final_engineered = engineered_data.copy()
    fe_config = params['feature_engineering']

    if not fe_config.get('create_interactions', False):
        return final_engineered

    log.info("🔧 Creating interaction features")
    non_poly_cols = [col for col in engineered_data.columns if 'poly' not in col]
    interaction_count = 0

    for i in range(len(non_poly_cols)):
        for j in range(i + 1, len(non_poly_cols)):
            col1, col2 = non_poly_cols[i], non_poly_cols[j]
            new_col = f"{col1}_x_{col2}"
            final_engineered[new_col] = engineered_data[col1] * engineered_data[col2]
            interaction_count += 1
            if interaction_count >= fe_config.get('max_interactions', 10):
                break
        if interaction_count >= fe_config.get('max_interactions', 10):
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
                inputs=["cleaned_data", "params:"],
                outputs="imputed_data",
                name="handle_missing_values_node",
                tags="fe",
            ),
            node(
                func=scale_features_node,
                inputs=["imputed_data", "params:"],
                outputs="scaled_data",
                name="scale_features_node",
                tags="fe",
            ),
            node(
                func=create_polynomial_features_node,
                inputs=["scaled_data", "params:"],
                outputs="polynomial_data",
                name="create_polynomial_features_node",
                tags="fe",
            ),
            node(
                func=create_interaction_features_node,
                inputs=["polynomial_data", "params:"],
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