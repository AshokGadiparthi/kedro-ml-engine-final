"""
Feature Engineering Pipeline - Phase 2
Handles missing values, scaling, encoding, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OneHotEncoder, LabelEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
import logging
from typing import Dict, Any

log = logging.getLogger(__name__)


def handle_missing_values_node(
        cleaned_data: pd.DataFrame,
        params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Handle missing values using configured strategy.

    Args:
        cleaned_data: Input DataFrame
        params: Pipeline parameters

    Returns:
        imputed_data: DataFrame with no missing values
    """
    strategy_config = params['missing_value_strategy']
    method = strategy_config['method']

    log.info(f"🔧 Imputing missing values using: {method}")

    if cleaned_data.isnull().sum().sum() == 0:
        log.info("✅ No missing values found, returning data as-is")
        return cleaned_data

    if method == 'mean':
        imputer = SimpleImputer(strategy='mean')
    elif method == 'median':
        imputer = SimpleImputer(strategy='median')
    elif method == 'forward_fill':
        limit = strategy_config.get('forward_fill_limit', 3)
        imputed = cleaned_data.fillna(method='ffill', limit=limit)
        log.info(f"✅ Forward filled with limit={limit}")
        return imputed
    elif method == 'knn':
        n_neighbors = strategy_config.get('knn_neighbors', 5)
        imputer = KNNImputer(n_neighbors=n_neighbors)
        log.info(f"KNN imputation with {n_neighbors} neighbors")
    else:
        log.warning(f"Unknown method: {method}, returning original")
        return cleaned_data

    imputed_array = imputer.fit_transform(cleaned_data)
    imputed_data = pd.DataFrame(imputed_array, columns=cleaned_data.columns)

    log.info(f"✅ Shape: {imputed_data.shape}")
    log.info(f"✅ Missing values: {imputed_data.isnull().sum().sum()}")

    return imputed_data


def scale_features_node(
        imputed_data: pd.DataFrame,
        params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Scale features using configured method.

    Args:
        imputed_data: Input DataFrame
        params: Pipeline parameters

    Returns:
        scaled_data: Scaled DataFrame
    """
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
        log.warning(f"Unknown method: {method}, returning original")
        return imputed_data

    scaled_array = scaler.fit_transform(imputed_data)
    scaled_data = pd.DataFrame(scaled_array, columns=imputed_data.columns)

    log.info(f"✅ Shape: {scaled_data.shape}")
    log.info(f"✅ Mean: {scaled_data.mean().mean():.4f}, Std: {scaled_data.std().mean():.4f}")

    return scaled_data


def create_polynomial_features_node(
        scaled_data: pd.DataFrame,
        params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Create polynomial features if configured.

    Args:
        scaled_data: Input DataFrame
        params: Pipeline parameters

    Returns:
        engineered_data: DataFrame with polynomial features
    """
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

    log.info(f"✅ Added polynomial features, shape: {engineered_data.shape}")
    return engineered_data


def create_interaction_features_node(
        engineered_data: pd.DataFrame,
        params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Create interaction features if configured.

    Args:
        engineered_data: Input DataFrame
        params: Pipeline parameters

    Returns:
        final_engineered: DataFrame with interaction features
    """
    final_engineered = engineered_data.copy()
    fe_config = params['feature_engineering']

    if not fe_config.get('create_interactions', False):
        return final_engineered

    log.info("🔧 Creating interaction features")

    # Get non-polynomial columns
    non_poly_cols = [col for col in engineered_data.columns if 'poly' not in col]
    interaction_count = 0

    for i in range(len(non_poly_cols)):
        for j in range(i + 1, len(non_poly_cols)):
            col1, col2 = non_poly_cols[i], non_poly_cols[j]
            new_col = f"{col1}_x_{col2}"
            final_engineered[new_col] = engineered_data[col1] * engineered_data[col2]
            interaction_count += 1

            if interaction_count >= fe_config.get('max_interactions', 10):
                log.info(f"⚠️ Reached max interactions ({interaction_count}), stopping")
                break
        if interaction_count >= fe_config.get('max_interactions', 10):
            break

    log.info(f"✅ Added {interaction_count} interactions, shape: {final_engineered.shape}")
    return final_engineered


def generate_feature_statistics_node(
        engineered_data: pd.DataFrame
) -> Dict[str, Any]:
    """
    Generate feature statistics.

    Args:
        engineered_data: Input DataFrame

    Returns:
        feature_stats: Dictionary of statistics
    """
    log.info("📊 Generating feature statistics")

    feature_stats = {
        'shape': list(engineered_data.shape),
        'columns': engineered_data.columns.tolist(),
        'n_features': engineered_data.shape[1],
        'n_samples': engineered_data.shape[0],
        'missing_values': engineered_data.isnull().sum().to_dict(),
        'data_types': {str(k): str(v) for k, v in engineered_data.dtypes.to_dict().items()},
        'numeric_stats': engineered_data.describe().to_dict(),
        'correlations': engineered_data.corr().to_dict(),
    }

    log.info(f"✅ Generated {len(feature_stats)} statistic sections")
    return feature_stats