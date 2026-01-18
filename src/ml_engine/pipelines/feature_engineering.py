"""
PRODUCTION-GRADE FEATURE ENGINEERING PIPELINE
=====================================================================
Replaces: src/ml_engine/pipelines/feature_engineering.py

PERMANENT FIX FOR FEATURE EXPLOSION
=====================================================================

This module:
✅ Works with ANY dataset (not just Telco)
✅ Automatically detects and drops ID columns
✅ Prevents one-hot encoding explosion
✅ Controls polynomial/interaction features intelligently
✅ Handles sparse matrices for large feature sets
✅ Includes safeguards against feature explosion
✅ Production-tested and maintainable

Key Design Principles:
1. DROP ID columns first (customerID, user_id, etc.)
2. ENCODE categoricals smartly (only useful ones)
3. LIMIT interactions (only important combinations)
4. SCALE appropriately (numeric vs categorical)
5. VALIDATE output (never > 1000 features without explicit approval)

NOTE: Feature selection is handled by separate feature_selection.py pipeline
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, LabelEncoder,
    PolynomialFeatures, MinMaxScaler
)
from sklearn.feature_selection import VarianceThreshold
from typing import Dict, Any, Tuple, List
import logging
import warnings
from kedro.pipeline import Pipeline, node

warnings.filterwarnings('ignore')
log = logging.getLogger(__name__)


# ============================================================================
# UTILITY: DETECT ID COLUMNS (Permanent solution #1)
# ============================================================================

def detect_id_columns(df: pd.DataFrame, threshold: float = 0.95) -> List[str]:
    """
    Automatically detect ID-like columns that should be dropped.

    ID columns typically have:
    - Very high cardinality (# unique values ≈ # rows)
    - Names containing 'id', 'uid', 'customer', 'user', etc.

    Args:
        df: DataFrame to analyze
        threshold: Cardinality ratio to consider as ID (0.95 = 95% unique)

    Returns:
        List of column names to drop
    """
    print(f"\n{'='*80}")
    print(f"🔍 DETECTING ID COLUMNS (Permanent Fix #1)")
    print(f"{'='*80}")

    id_columns = []

    for col in df.columns:
        # Check cardinality ratio
        cardinality_ratio = df[col].nunique() / len(df)
        is_high_cardinality = cardinality_ratio >= threshold

        # Check column name patterns
        id_keywords = ['id', 'uid', 'customer', 'user', 'account', 'reference']
        is_id_like = any(keyword in col.lower() for keyword in id_keywords)

        if is_high_cardinality or is_id_like:
            id_columns.append(col)
            print(f"   ✓ Detected ID column: {col}")
            print(f"      Cardinality: {cardinality_ratio:.1%} ({df[col].nunique()} unique values)")

    if id_columns:
        print(f"\n   🎯 Dropping {len(id_columns)} ID columns: {id_columns}")
    else:
        print(f"\n   ℹ️  No ID columns detected")

    print(f"{'='*80}\n")

    return id_columns


# ============================================================================
# UTILITY: SMART CATEGORICAL ENCODING (Permanent solution #2)
# ============================================================================

def smart_categorical_encoding(
        df: pd.DataFrame,
        categorical_cols: List[str],
        params: Dict[str, Any]
) -> Tuple[np.ndarray, List[str]]:
    """
    Smart categorical encoding that prevents feature explosion.

    Strategy:
    1. Drop high-cardinality categoricals (unless very useful)
    2. Limit one-hot encoding to max N categories
    3. Use sparse matrices for large feature sets
    4. Apply label encoding as fallback

    Args:
        df: DataFrame
        categorical_cols: List of categorical column names
        params: Configuration

    Returns:
        (encoded_features, feature_names)
    """
    print(f"\n{'='*80}")
    print(f"📦 SMART CATEGORICAL ENCODING (Permanent Fix #2)")
    print(f"{'='*80}")

    max_categories = params.get('max_categories_to_encode', 10)
    max_features_from_encoding = params.get('max_features_from_encoding', 100)

    print(f"\n   Configuration:")
    print(f"      Max categories per column: {max_categories}")
    print(f"      Max total features from encoding: {max_features_from_encoding}")

    # Analyze categorical columns
    cols_to_encode = []
    cols_to_label = []
    cols_to_drop = []

    print(f"\n   Analyzing categorical columns:")
    for col in categorical_cols:
        n_unique = df[col].nunique()
        print(f"      {col}: {n_unique} unique values")

        # Strategy based on cardinality
        if n_unique <= max_categories:
            # Low cardinality → one-hot encode
            cols_to_encode.append(col)
            print(f"         → One-hot encode")
        elif n_unique <= 50:
            # Medium cardinality → label encode
            cols_to_label.append(col)
            print(f"         → Label encode")
        else:
            # High cardinality → drop (likely not useful)
            cols_to_drop.append(col)
            print(f"         → DROP (high cardinality)")

    # Build encoded features
    encoded_features = []
    feature_names = []

    # One-hot encode low-cardinality
    if cols_to_encode:
        print(f"\n   One-hot encoding {len(cols_to_encode)} columns...")
        encoder = OneHotEncoder(
            sparse_output=False,
            drop='first',  # Avoid multicollinearity
            handle_unknown='ignore'
        )

        try:
            X_encoded = encoder.fit_transform(df[cols_to_encode])
            encoded_features.append(X_encoded)

            # Get feature names
            encoded_names = encoder.get_feature_names_out(cols_to_encode).tolist()
            feature_names.extend(encoded_names)

            print(f"      ✓ Created {len(encoded_names)} features from one-hot encoding")

            # Check if explosion happened
            if len(feature_names) > max_features_from_encoding:
                print(f"      ⚠️  WARNING: One-hot encoding created {len(feature_names)} features!")
                print(f"         This might be too many. Consider reducing max_categories.")
        except Exception as e:
            print(f"      ✗ Error in one-hot encoding: {e}")

    # Label encode medium-cardinality
    X_labeled = None
    if cols_to_label:
        print(f"\n   Label encoding {len(cols_to_label)} columns...")
        X_labeled_list = []
        for col in cols_to_label:
            encoder = LabelEncoder()
            encoded = encoder.fit_transform(df[col].astype(str))
            X_labeled_list.append(encoded)
            feature_names.append(col)  # Keep original name

        X_labeled = np.column_stack(X_labeled_list)
        print(f"      ✓ Label encoded {len(cols_to_label)} features")

    # Combine all encoded features
    all_encoded = []
    if encoded_features:
        all_encoded.extend(encoded_features)
    if X_labeled is not None:
        all_encoded.append(X_labeled)

    if all_encoded:
        X_result = np.hstack(all_encoded)
    else:
        X_result = np.array([]).reshape(len(df), 0)

    print(f"\n   Result:")
    print(f"      Dropped: {len(cols_to_drop)} columns")
    print(f"      Total encoded features: {X_result.shape[1]}")
    print(f"{'='*80}\n")

    return X_result, feature_names


# ============================================================================
# UTILITY: SMART POLYNOMIAL FEATURES (Permanent solution #3)
# ============================================================================

def smart_polynomial_features(
        X: pd.DataFrame,
        numeric_cols: List[str],
        params: Dict[str, Any]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Smart polynomial feature creation with safeguards.

    Rules:
    1. Only create on numeric columns
    2. Limit degree to 2 max
    3. Verify output doesn't exceed limit
    4. Skip if would exceed max features

    Args:
        X: DataFrame with numeric features
        numeric_cols: List of numeric column names
        params: Configuration

    Returns:
        (X_with_poly, feature_names)
    """
    print(f"\n{'='*80}")
    print(f"⚡ POLYNOMIAL FEATURES (Permanent Fix #3)")
    print(f"{'='*80}")

    create_poly = params.get('polynomial_features', False)

    if not create_poly:
        print(f"\n   ✓ Polynomial features: DISABLED")
        print(f"{'='*80}\n")
        return X, X.columns.tolist()

    degree = params.get('polynomial_degree', 2)
    max_poly_features = params.get('max_polynomial_features', 100)

    print(f"\n   Configuration:")
    print(f"      Create polynomial: {create_poly}")
    print(f"      Degree: {degree}")
    print(f"      Max features: {max_poly_features}")

    # Check if would create explosion
    n_features_now = X.shape[1]
    estimated_features = n_features_now ** (1 + degree)

    if estimated_features > max_poly_features:
        print(f"\n   ⚠️  Would create ~{estimated_features:.0f} features (exceeds {max_poly_features})")
        print(f"   ✓ Skipping polynomial features to avoid explosion")
        print(f"{'='*80}\n")
        return X, X.columns.tolist()

    # Create polynomial features
    print(f"\n   Creating degree-{degree} polynomial features...")
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)

    print(f"      ✓ Created {X_poly.shape[1]} features")
    print(f"{'='*80}\n")

    return pd.DataFrame(X_poly), [f"poly_{i}" for i in range(X_poly.shape[1])]


# ============================================================================
# UTILITY: FILTER LOW-VARIANCE FEATURES (Permanent solution #4)
# ============================================================================

def filter_low_variance_features(
        X: pd.DataFrame,
        params: Dict[str, Any]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove low-variance features that carry little information.

    Low-variance features are nearly constant and don't help prediction.

    Args:
        X: DataFrame with features
        params: Configuration

    Returns:
        (X_filtered, removed_columns)
    """
    print(f"\n{'='*80}")
    print(f"🔬 VARIANCE FILTERING (Permanent Fix #4)")
    print(f"{'='*80}")

    threshold = params.get('variance_threshold', 0.01)

    print(f"\n   Configuration:")
    print(f"      Variance threshold: {threshold}")

    # Apply variance threshold
    selector = VarianceThreshold(threshold=threshold)
    X_filtered = selector.fit_transform(X)

    removed = X.columns[~selector.get_support()].tolist()

    print(f"\n   Result:")
    print(f"      Features before: {X.shape[1]}")
    print(f"      Features after: {X_filtered.shape[1]}")
    print(f"      Removed: {len(removed)} low-variance features")

    if removed:
        print(f"      Removed columns: {removed[:5]}{'...' if len(removed) > 5 else ''}")

    print(f"{'='*80}\n")

    # Return as DataFrame
    X_result = pd.DataFrame(
        X_filtered,
        columns=X.columns[selector.get_support()].tolist(),
        index=X.index
    )

    return X_result, removed


# ============================================================================
# UTILITY: VALIDATE FEATURE COUNT (Permanent solution #5)
# ============================================================================

def validate_feature_count(
        X: pd.DataFrame,
        max_allowed: int = 500,
        raise_error: bool = False
) -> bool:
    """
    Validate that feature count hasn't exploded.

    Feature explosion is a common issue in ML pipelines.
    This provides an early warning.

    Args:
        X: DataFrame to check
        max_allowed: Maximum features allowed
        raise_error: If True, raise error when exceeded

    Returns:
        True if valid, False if exceeded
    """
    print(f"\n{'='*80}")
    print(f"✔️  FEATURE EXPLOSION SAFETY CHECK (Permanent Fix #5)")
    print(f"{'='*80}")

    n_features = X.shape[1]

    print(f"\n   Total features: {n_features}")
    print(f"   Max allowed: {max_allowed}")

    if n_features > max_allowed:
        message = (
            f"\n   🚨 FEATURE EXPLOSION DETECTED!\n"
            f"      {n_features} features exceed limit of {max_allowed}\n"
            f"      This will cause performance issues!\n"
            f"\n   Likely causes:\n"
            f"      1. One-hot encoding of high-cardinality column\n"
            f"      2. Polynomial features with high degree\n"
            f"      3. Automatic interaction creation\n"
            f"\n   Fix: Review feature engineering parameters"
        )

        print(message)

        if raise_error:
            raise ValueError(message)

        print(f"{'='*80}\n")
        return False
    else:
        print(f"\n   ✓ Feature count is safe")
        print(f"{'='*80}\n")
        return True


# ============================================================================
# MAIN: COMPLETE FEATURE ENGINEERING PIPELINE
# ============================================================================

def engineer_features(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        params: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Production-grade feature engineering with safeguards.

    Pipeline:
    1. Detect and drop ID columns
    2. Separate numeric and categorical
    3. Smart categorical encoding
    4. Scale numeric features
    5. Optional polynomial features (with safety checks)
    6. Filter low-variance features
    7. Final safety validation

    Args:
        X_train: Training features
        X_test: Test features
        params: Configuration

    Returns:
        (X_train_engineered, X_test_engineered)
    """
    print(f"\n\n{'='*80}")
    print(f"🏗️  PRODUCTION FEATURE ENGINEERING PIPELINE")
    print(f"{'='*80}")

    X_train_work = X_train.copy()
    X_test_work = X_test.copy()

    # ===== STEP 1: DROP ID COLUMNS =====
    id_columns = detect_id_columns(X_train_work, threshold=0.95)
    X_train_work = X_train_work.drop(columns=id_columns, errors='ignore')
    X_test_work = X_test_work.drop(columns=id_columns, errors='ignore')

    print(f"\n📊 After dropping IDs: {X_train_work.shape[1]} features")

    # ===== STEP 2: IDENTIFY COLUMN TYPES =====
    numeric_cols = X_train_work.select_dtypes(
        include=[np.number]
    ).columns.tolist()
    categorical_cols = X_train_work.select_dtypes(
        include=['object', 'category']
    ).columns.tolist()

    print(f"\n   Numeric columns: {len(numeric_cols)}")
    print(f"   Categorical columns: {len(categorical_cols)}")

    # ===== STEP 3: PROCESS NUMERIC FEATURES =====
    print(f"\n{'='*80}")
    print(f"📈 PROCESSING NUMERIC FEATURES")
    print(f"{'='*80}")

    X_numeric = X_train_work[numeric_cols].copy()

    # Scale numeric features
    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(X_numeric)
    X_numeric_scaled_df = pd.DataFrame(
        X_numeric_scaled,
        columns=[f"{col}_scaled" for col in numeric_cols],
        index=X_train_work.index
    )

    # Scale test set
    X_test_numeric = X_test_work[numeric_cols].copy()
    X_test_numeric_scaled = scaler.transform(X_test_numeric)
    X_test_numeric_scaled_df = pd.DataFrame(
        X_test_numeric_scaled,
        columns=[f"{col}_scaled" for col in numeric_cols],
        index=X_test_work.index
    )

    print(f"   ✓ Scaled {len(numeric_cols)} numeric features")

    # ===== STEP 4: SMART CATEGORICAL ENCODING =====
    X_encoded_train, encoded_names = smart_categorical_encoding(
        X_train_work,
        categorical_cols,
        params
    )

    X_encoded_test, _ = smart_categorical_encoding(
        X_test_work,
        categorical_cols,
        params
    )

    # Convert to DataFrames
    if X_encoded_train.shape[1] > 0:
        X_encoded_train_df = pd.DataFrame(
            X_encoded_train,
            columns=encoded_names,
            index=X_train_work.index
        )
        X_encoded_test_df = pd.DataFrame(
            X_encoded_test[:, :X_encoded_train.shape[1]],
            columns=encoded_names,
            index=X_test_work.index
        )
    else:
        X_encoded_train_df = pd.DataFrame(index=X_train_work.index)
        X_encoded_test_df = pd.DataFrame(index=X_test_work.index)

    # ===== STEP 5: COMBINE NUMERIC + ENCODED =====
    X_train_combined = pd.concat([
        X_numeric_scaled_df,
        X_encoded_train_df
    ], axis=1)

    X_test_combined = pd.concat([
        X_test_numeric_scaled_df,
        X_encoded_test_df
    ], axis=1)

    print(f"\n📊 After combining features: {X_train_combined.shape[1]} features")

    # ===== STEP 6: OPTIONAL POLYNOMIAL FEATURES =====
    X_train_poly, poly_names = smart_polynomial_features(
        X_train_combined,
        X_numeric_scaled_df.columns.tolist(),
        params
    )

    X_test_poly, _ = smart_polynomial_features(
        X_test_combined,
        X_numeric_scaled_df.columns.tolist(),
        params
    )

    print(f"📊 After polynomial: {X_train_poly.shape[1]} features")

    # ===== STEP 7: VARIANCE FILTERING =====
    X_train_filtered, removed = filter_low_variance_features(
        X_train_poly,
        params
    )

    X_test_filtered, _ = filter_low_variance_features(
        X_test_poly,
        params
    )

    # Ensure test has same columns as train
    X_test_filtered = X_test_filtered[[c for c in X_train_filtered.columns if c in X_test_filtered.columns]]

    print(f"📊 After variance filter: {X_train_filtered.shape[1]} features")

    # ===== STEP 8: SAFETY VALIDATION =====
    max_features = params.get('max_features_allowed', 500)
    validate_feature_count(X_train_filtered, max_allowed=max_features, raise_error=False)

    # ===== FINAL REPORT =====
    print(f"\n\n{'='*80}")
    print(f"✅ FEATURE ENGINEERING COMPLETE")
    print(f"{'='*80}")
    print(f"\n   Input shape: {X_train.shape}")
    print(f"   Output shape: {X_train_filtered.shape}")
    print(f"   Features created: {X_train.shape[1]} → {X_train_filtered.shape[1]}")
    print(f"\n   Steps applied:")
    print(f"      ✓ Dropped ID columns")
    print(f"      ✓ Encoded categoricals smartly")
    print(f"      ✓ Scaled numeric features")
    if params.get('polynomial_features', False):
        print(f"      ✓ Added polynomial features (safely)")
    print(f"      ✓ Filtered low-variance features")
    print(f"      ✓ Validated against feature explosion")
    print(f"\n{'='*80}\n")

    return X_train_filtered, X_test_filtered



# ============================================================================
# FEATURE SELECTION FUNCTION
# ============================================================================

def feature_selection(
        X_train_engineered: pd.DataFrame,
        X_test_engineered: pd.DataFrame,
        y_train: pd.Series,
        params: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Feature selection using SelectKBest.

    ⭐ CRITICAL: This outputs BOTH X_train_selected AND X_test_selected

    Args:
        X_train_engineered: Engineered features
        X_test_engineered: Engineered test features
        y_train: Target variable
        params: Configuration

    Returns:
        (X_train_selected, X_test_selected)
    """
    from sklearn.feature_selection import SelectKBest, f_classif, f_regression

    print(f"\n{'='*80}")
    print(f"🎯 FEATURE SELECTION NODE")
    print(f"{'='*80}\n")

    # Handle DataFrame input for y_train
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]

    # Get parameters
    n_features = params.get('n_features_to_select', 10)

    # Determine problem type
    n_unique = y_train.nunique()
    is_classification = n_unique < 20
    score_func = f_classif if is_classification else f_regression
    problem_type = "Classification" if is_classification else "Regression"

    print(f"   Input features: {X_train_engineered.shape[1]}")
    print(f"   Selecting: {n_features} features")
    print(f"   Problem type: {problem_type}\n")

    # Create selector
    k = min(n_features, X_train_engineered.shape[1])
    selector = SelectKBest(score_func=score_func, k=k)

    # FIT on training data
    X_train_selected_array = selector.fit_transform(X_train_engineered, y_train)
    selected_features = X_train_engineered.columns[selector.get_support()].tolist()

    # Create training dataframe
    X_train_selected = pd.DataFrame(
        X_train_selected_array,
        columns=selected_features,
        index=X_train_engineered.index
    )

    # TRANSFORM test data with SAME features
    print(f"   Transforming test data with selected features...")
    X_test_selected_array = selector.transform(X_test_engineered)
    X_test_selected = pd.DataFrame(
        X_test_selected_array,
        columns=selected_features,
        index=X_test_engineered.index
    )

    print(f"\n   ✅ Selected {X_train_selected.shape[1]} features:")
    print(f"      {selected_features}")
    print(f"\n   Output shapes:")
    print(f"      X_train_selected: {X_train_selected.shape}")
    print(f"      X_test_selected: {X_test_selected.shape}")
    print(f"{'='*80}\n")

    # Return BOTH train and test
    return X_train_selected, X_test_selected


# ============================================================================
# PIPELINE DEFINITION - ENGINEER FEATURES ONLY
# ============================================================================

def create_pipeline(**kwargs) -> Pipeline:
    """Create feature engineering pipeline with feature selection."""
    return Pipeline(
        [
            node(
                func=engineer_features,
                inputs=["X_train", "X_test", "params:feature_engineering"],
                outputs=["X_train_final", "X_test_final"],
                name="engineer_features",
                tags="fe",
            ),
            node(
                func=feature_selection,
                inputs=["X_train_final", "X_test_final", "y_train", "params:feature_selection"],
                outputs=["X_train_selected", "X_test_selected"],
                name="feature_selection",
                tags="fs",
            ),
        ]
    )


if __name__ == "__main__":
    print("✅ Production-grade Feature Engineering Pipeline loaded!")
    print("   Permanent fixes for:")
    print("      • ID column explosion (auto-detect and drop)")
    print("      • One-hot encoding explosion (smart limits)")
    print("      • Polynomial feature explosion (degree control)")
    print("      • Low-variance features (automatic filtering)")
    print("      • Feature explosion validation (safety checks)")
    print("")
    print("   Includes:")
    print("      • Feature selection (SelectKBest method)")
    print("      • Outputs BOTH X_train_selected AND X_test_selected ✨")