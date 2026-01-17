"""
PERFECT PHASE 2 - FEATURE ENGINEERING
=====================================================================
Replaces: src/ml_engine/pipelines/feature_engineering.py

Handles:
  ✅ Separate numeric and categorical features (Gap 4)
  ✅ Fit preprocessors on TRAIN only (Gap 1)
  ✅ Transform both train AND test with fitted preprocessors
  ✅ Prevent feature explosion (Gap 3)
  ✅ Multiple encoding strategies for categorical data

The key principle: FIT on train, TRANSFORM both
=====================================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import logging
from typing import Dict, Any, Tuple, Optional
from kedro.pipeline import Pipeline, node
import joblib
from pathlib import Path

log = logging.getLogger(__name__)


# ============================================================================
# PREPROCESSING PIPELINE - FIT ON TRAIN, TRANSFORM BOTH
# ============================================================================

def handle_missing_values_train(
        X_train: pd.DataFrame,
        params: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fit missing value imputer on TRAINING data ONLY.

    Args:
        X_train: Training features
        params: Configuration

    Returns:
        (X_train_imputed, imputer_config)
    """
    print(f"\n🔧 FITTING Imputer on TRAINING data")

    if X_train.isnull().sum().sum() == 0:
        log.info("✅ No missing values in training data")
        return X_train, {'method': 'none'}

    strategy_config = params.get('missing_value_strategy', {})
    method = strategy_config.get('method', 'mean') if isinstance(strategy_config, dict) else strategy_config

    print(f"   Method: {method}")
    print(f"   Missing values: {X_train.isnull().sum().sum()}")

    # Separate numeric and categorical
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    X_train_imputed = X_train.copy()
    imputer_config = {'method': method, 'numeric_cols': numeric_cols}

    if numeric_cols:
        if method == 'mean':
            imputer = SimpleImputer(strategy='mean')
        elif method == 'median':
            imputer = SimpleImputer(strategy='median')
        elif method == 'knn':
            imputer = KNNImputer(n_neighbors=5)
        else:
            imputer = SimpleImputer(strategy='mean')

        # FIT on training data ONLY
        imputer.fit(X_train[numeric_cols])
        X_train_imputed[numeric_cols] = imputer.transform(X_train[numeric_cols])
        imputer_config['imputer'] = imputer

    print(f"   ✅ Imputer fitted on {len(numeric_cols)} numeric features")
    return X_train_imputed, imputer_config


def handle_missing_values_test(
        X_test: pd.DataFrame,
        imputer_config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Transform test data using FITTED imputer.

    Args:
        X_test: Test features
        imputer_config: Config from training

    Returns:
        X_test imputed
    """
    print(f"\n🔄 TRANSFORMING test data with fitted imputer")

    X_test_imputed = X_test.copy()

    if imputer_config.get('imputer'):
        imputer = imputer_config['imputer']
        numeric_cols = imputer_config['numeric_cols']
        X_test_imputed[numeric_cols] = imputer.transform(X_test[numeric_cols])
        print(f"   ✅ Applied imputation to {len(numeric_cols)} features")
    else:
        print(f"   ℹ️  No imputation needed")

    return X_test_imputed


def scale_features_train(
        X_train: pd.DataFrame,
        params: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fit scaler on TRAINING data ONLY.

    Args:
        X_train: Training features
        params: Configuration

    Returns:
        (X_train_scaled, scaler_config)
    """
    print(f"\n🔧 FITTING Scaler on TRAINING data")

    scaling_config = params.get('scaling', {})
    method = scaling_config.get('method', 'standard') if isinstance(scaling_config, dict) else scaling_config

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        return X_train, {'method': 'none'}

    print(f"   Method: {method}")
    print(f"   Features: {len(numeric_cols)}")

    X_train_scaled = X_train.copy()

    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler(feature_range=(0, 1))
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        return X_train, {'method': 'none'}

    # FIT on training data ONLY
    scaler.fit(X_train[numeric_cols])
    X_train_scaled[numeric_cols] = scaler.transform(X_train[numeric_cols])

    print(f"   ✅ Scaler fitted on training statistics")

    return X_train_scaled, {
        'method': method,
        'scaler': scaler,
        'numeric_cols': numeric_cols
    }


def scale_features_test(
        X_test: pd.DataFrame,
        scaler_config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Transform test data using FITTED scaler.

    Args:
        X_test: Test features
        scaler_config: Config from training

    Returns:
        X_test scaled
    """
    print(f"\n🔄 TRANSFORMING test data with fitted scaler")

    X_test_scaled = X_test.copy()

    if scaler_config.get('scaler'):
        scaler = scaler_config['scaler']
        numeric_cols = scaler_config['numeric_cols']
        X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
        print(f"   ✅ Applied {scaler_config['method']} scaling")
    else:
        print(f"   ℹ️  No scaling applied")

    return X_test_scaled


def encode_categorical_train(
        X_train: pd.DataFrame,
        params: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fit categorical encoders on TRAINING data ONLY.

    Args:
        X_train: Training features
        params: Configuration

    Returns:
        (X_train_encoded, encoder_config)
    """
    print(f"\n🔧 FITTING Categorical Encoders on TRAINING data")

    categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    if not categorical_cols:
        print(f"   ℹ️  No categorical features")
        return X_train, {'method': 'none', 'categorical_cols': []}

    encoding_method = params.get('categorical_encoding', 'onehot')

    print(f"   Method: {encoding_method}")
    print(f"   Categorical columns: {len(categorical_cols)}")
    print(f"   Columns: {categorical_cols}")

    X_train_encoded = X_train.copy()
    encoder_config = {
        'method': encoding_method,
        'categorical_cols': categorical_cols
    }

    if encoding_method == 'onehot':
        # One-hot encoding
        X_train_encoded = pd.get_dummies(
            X_train_encoded,
            columns=categorical_cols,
            drop_first=True
        )
        encoder_config['encoded_columns'] = X_train_encoded.columns.tolist()
        print(f"   ✅ One-hot encoded: created {len(X_train_encoded.columns) - len(X_train.columns) + len(categorical_cols)} dummy variables")

    elif encoding_method == 'label':
        # Label encoding
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            le.fit(X_train[col].astype(str))
            X_train_encoded[col] = le.transform(X_train[col].astype(str))
            encoders[col] = le
        encoder_config['encoders'] = encoders
        print(f"   ✅ Label encoded {len(categorical_cols)} features")

    return X_train_encoded, encoder_config


def encode_categorical_test(
        X_test: pd.DataFrame,
        encoder_config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Transform test data using FITTED encoders.

    Args:
        X_test: Test features
        encoder_config: Config from training

    Returns:
        X_test encoded
    """
    print(f"\n🔄 TRANSFORMING test data with fitted encoders")

    X_test_encoded = X_test.copy()
    categorical_cols = encoder_config.get('categorical_cols', [])

    if not categorical_cols:
        print(f"   ℹ️  No categorical encoding")
        return X_test

    method = encoder_config.get('method', 'onehot')

    if method == 'onehot':
        encoded_columns = encoder_config.get('encoded_columns', [])
        X_test_encoded = pd.get_dummies(
            X_test_encoded,
            columns=categorical_cols,
            drop_first=True
        )

        # Ensure same columns as training
        missing_cols = set(encoded_columns) - set(X_test_encoded.columns)
        for col in missing_cols:
            X_test_encoded[col] = 0

        # Keep only training columns
        X_test_encoded = X_test_encoded[encoded_columns]
        print(f"   ✅ One-hot encoded with {len(encoded_columns)} columns")

    elif method == 'label':
        encoders = encoder_config.get('encoders', {})
        for col, le in encoders.items():
            X_test_encoded[col] = le.transform(X_test[col].astype(str))
        print(f"   ✅ Label encoded {len(categorical_cols)} features")

    return X_test_encoded


def guard_polynomial_explosion(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        params: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Guard against feature explosion from polynomials (Gap 3 Fix).

    Args:
        X_train, X_test: Training and test features
        params: Configuration

    Returns:
        (X_train, X_test, poly_config)
    """
    print(f"\n🛡️  FEATURE EXPLOSION GUARD")

    poly_degree = params.get('polynomial_degree', 2)
    max_output_features = params.get('max_output_features', 1000)

    if poly_degree <= 1:
        print(f"   ℹ️  Polynomial disabled (degree={poly_degree})")
        return X_train, X_test, {'degree': 0}

    n_numeric = X_train.select_dtypes(include=[np.number]).shape[1]

    # Project output features
    if poly_degree == 2:
        projected = n_numeric * (n_numeric + 1) // 2
    else:
        projected = min(n_numeric ** poly_degree, 10000)  # Cap estimate

    print(f"   Input features: {n_numeric}")
    print(f"   Polynomial degree: {poly_degree}")
    print(f"   Projected output: {projected:,}")
    print(f"   Max allowed: {max_output_features:,}")

    if projected > max_output_features:
        print(f"\n   ⚠️  EXPLOSION DETECTED! Would create {projected:,} features")
        print(f"   ✓ Solution: Skipping polynomial features")
        print(f"   ✓ Keep features as-is to prevent memory crash")
        return X_train, X_test, {'degree': 0, 'skipped': True}

    # Create polynomials safely
    print(f"\n   ✓ Safe to create polynomials")
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols[:10]:  # Limit to first 10 to be safe
        for d in range(2, poly_degree + 1):
            X_train[f"{col}_poly{d}"] = X_train[col] ** d
            X_test[f"{col}_poly{d}"] = X_test[col] ** d

    print(f"   ✅ Created polynomial features")

    return X_train, X_test, {
        'degree': poly_degree,
        'numeric_cols': numeric_cols[:10]
    }


def log_feature_engineering_summary(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
) -> Dict[str, Any]:
    """Log summary of feature engineering."""
    summary = {
        'train_shape': X_train.shape,
        'test_shape': X_test.shape,
        'feature_count': X_train.shape[1],
        'numeric_count': X_train.select_dtypes(include=[np.number]).shape[1],
        'categorical_count': X_train.select_dtypes(exclude=[np.number]).shape[1],
    }

    log.info(f"\n📊 Feature Engineering Summary:")
    log.info(f"   Train shape: {summary['train_shape']}")
    log.info(f"   Test shape: {summary['test_shape']}")
    log.info(f"   Total features: {summary['feature_count']}")

    return summary


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create feature engineering pipeline.

    Proper flow:
    1. Fit preprocessors on TRAIN
    2. Transform both TRAIN and TEST
    3. Combine results
    """
    return Pipeline(
        [
            # Imputation
            node(
                func=handle_missing_values_train,
                inputs=["X_train_raw", "params:feature_engineering"],
                outputs=["X_train_imputed", "imputer_config"],
                name="handle_missing_values_train",
                tags="fe",
            ),
            node(
                func=handle_missing_values_test,
                inputs=["X_test_raw", "imputer_config"],
                outputs="X_test_imputed",
                name="handle_missing_values_test",
                tags="fe",
            ),
            # Scaling
            node(
                func=scale_features_train,
                inputs=["X_train_imputed", "params:feature_engineering"],
                outputs=["X_train_scaled", "scaler_config"],
                name="scale_features_train",
                tags="fe",
            ),
            node(
                func=scale_features_test,
                inputs=["X_test_imputed", "scaler_config"],
                outputs="X_test_scaled",
                name="scale_features_test",
                tags="fe",
            ),
            # Categorical Encoding
            node(
                func=encode_categorical_train,
                inputs=["X_train_scaled", "params:feature_engineering"],
                outputs=["X_train_encoded", "encoder_config"],
                name="encode_categorical_train",
                tags="fe",
            ),
            node(
                func=encode_categorical_test,
                inputs=["X_test_scaled", "encoder_config"],
                outputs="X_test_encoded",
                name="encode_categorical_test",
                tags="fe",
            ),
            # Feature Explosion Guard
            node(
                func=guard_polynomial_explosion,
                inputs=["X_train_encoded", "X_test_encoded", "params:feature_engineering"],
                outputs=["X_train_final", "X_test_final", "poly_config"],
                name="guard_polynomial_explosion",
                tags="fe",
            ),
            # Summary
            node(
                func=log_feature_engineering_summary,
                inputs=["X_train_final", "X_test_final"],
                outputs="fe_summary",
                name="log_feature_engineering_summary",
                tags="fe",
            ),
        ]
    )


if __name__ == "__main__":
    print("✅ Perfect Phase 2 Feature Engineering pipeline created!")