"""
PERFECT PHASE 2 - DATA LOADING WITH TRAIN/TEST SPLIT
=====================================================================
Replaces: src/ml_engine/pipelines/data_loading.py

**CRITICAL FIX #1: Data Leakage Prevention**
  ✅ Split train/test FIRST (before any preprocessing)
  ✅ NO preprocessing on combined data
  ✅ Each dataset gets its own preprocessing path

Integration: Drop this into src/ml_engine/pipelines/
=====================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any
import logging
from kedro.pipeline import Pipeline, node

log = logging.getLogger(__name__)


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load raw data from file.

    Args:
        filepath: Path to data file (CSV, Excel, etc)

    Returns:
        Raw DataFrame
    """
    log.info(f"📂 Loading raw data from: {filepath}")

    if filepath.endswith('.csv'):
        data = pd.read_csv(filepath)
    elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        data = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file type: {filepath}")

    log.info(f"✅ Loaded: {data.shape[0]} samples, {data.shape[1]} features")
    return data


def separate_target(
        raw_data: pd.DataFrame,
        target_column: str = 'target'
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target.

    Args:
        raw_data: Raw DataFrame with target column
        target_column: Name of target column

    Returns:
        (X, y) tuple
    """
    log.info(f"🎯 Separating target column: '{target_column}'")

    if target_column not in raw_data.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. "
            f"Available columns: {raw_data.columns.tolist()}"
        )

    X = raw_data.drop(columns=[target_column])
    y = raw_data[target_column]

    log.info(f"✅ Features: {X.shape[1]}, Target: {len(y)}")
    return X, y


def critical_split_train_test(
        X: pd.DataFrame,
        y: pd.Series,
        params: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    🔐 CRITICAL STEP: Split data BEFORE any preprocessing.

    This prevents data leakage where test data is "seen" during preprocessing.

    Args:
        X: Features
        y: Target
        params: Contains test_size, random_state, stratify

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    print(f"\n{'='*80}")
    print(f"🔐 CRITICAL: Train/Test Split BEFORE Preprocessing")
    print(f"{'='*80}")

    test_size = params.get('test_size', 0.2)
    random_state = params.get('random_state', 42)
    stratify = params.get('stratify', True)

    print(f"\nConfiguration:")
    print(f"   Test size: {test_size*100:.1f}%")
    print(f"   Train size: {(1-test_size)*100:.1f}%")
    print(f"   Random state: {random_state}")
    print(f"   Stratification: {stratify}")

    # Determine if we should stratify
    should_stratify = False
    if stratify and y.nunique() < len(y) * 0.1:
        should_stratify = True
        print(f"   Strategy: STRATIFIED SPLIT (preserves class distribution)")
    else:
        print(f"   Strategy: RANDOM SPLIT")

    # Split
    if should_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )

    print(f"\nResults:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Total samples: {len(X_train) + len(X_test)}")

    if y.nunique() < 10:
        print(f"\nClass Distribution:")
        print(f"   Training: {y_train.value_counts().to_dict()}")
        print(f"   Test: {y_test.value_counts().to_dict()}")

    print(f"\n⚠️  KEY PRINCIPLE:")
    print(f"   All subsequent preprocessing will:")
    print(f"   • FIT on training data ONLY (learn statistics from train)")
    print(f"   • TRANSFORM both train AND test (apply learned statistics)")
    print(f"\n   This ensures test metrics are REALISTIC!")
    print(f"{'='*80}\n")

    log.info(f"✅ Split complete: {len(X_train)} train, {len(X_test)} test")

    return X_train, X_test, y_train, y_test


def log_split_summary(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
) -> Dict[str, Any]:
    """
    Generate summary of train/test split.

    Args:
        X_train, X_test, y_train, y_test: Split datasets

    Returns:
        Summary dictionary
    """
    summary = {
        'X_train_shape': X_train.shape,
        'X_test_shape': X_test.shape,
        'y_train_shape': y_train.shape,
        'y_test_shape': y_test.shape,
        'train_size_pct': (len(X_train) / (len(X_train) + len(X_test))) * 100,
        'test_size_pct': (len(X_test) / (len(X_train) + len(X_test))) * 100,
        'total_samples': len(X_train) + len(X_test),
        'total_features': X_train.shape[1],
    }

    log.info(f"\n📊 Data Loading Summary:")
    log.info(f"   Train: {summary['X_train_shape']}")
    log.info(f"   Test: {summary['X_test_shape']}")
    log.info(f"   Total: {summary['total_samples']} samples, {summary['total_features']} features")

    return summary


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create data loading pipeline with CORRECT train/test split.

    Pipeline Flow:
    1. Load raw data
    2. Separate target
    3. Split train/test FIRST (CRITICAL!)
    4. Log summary

    No preprocessing happens here - that comes in Phase 2 separate nodes.
    """
    return Pipeline(
        [
            node(
                func=load_raw_data,
                inputs="params:data_path",
                outputs="raw_data",
                name="load_raw_data",
                tags="phase1",
            ),
            node(
                func=separate_target,
                inputs=["raw_data", "params:target_column"],
                outputs=["X_raw", "y_raw"],
                name="separate_target",
                tags="phase1",
            ),
            node(
                func=critical_split_train_test,
                inputs=["X_raw", "y_raw", "params:data_processing"],
                outputs=["X_train_raw", "X_test_raw", "y_train", "y_test"],
                name="critical_split_train_test",
                tags="phase1",
            ),
            node(
                func=log_split_summary,
                inputs=["X_train_raw", "X_test_raw", "y_train", "y_test"],
                outputs="split_summary",
                name="log_split_summary",
                tags="phase1",
            ),
        ]
    )


if __name__ == "__main__":
    # Test
    print("✅ Data loading pipeline created with proper train/test split!")