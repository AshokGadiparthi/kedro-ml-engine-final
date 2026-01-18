"""
PHASE 3: MODEL TRAINING & EVALUATION WITH PATH B FEATURE SCALING & ADVANCED TUNING
================================================================================
UPDATED: Added feature scaling + advanced hyperparameter tuning + cross-validation
Inputs: X_train_selected, X_test_selected, y_train, y_test (ONLY)
Outputs: baseline_model, best_model, model_evaluation, phase3_predictions, scalers
================================================================================
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple
import pickle
import json
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, classification_report,
    roc_auc_score, roc_curve, confusion_matrix, auc
)
from kedro.pipeline import Pipeline, node
import matplotlib.pyplot as plt
import seaborn as sns

log = logging.getLogger(__name__)


# ============================================================================
# PHASE 3.0: FEATURE SCALING (PATH B - NEW)
# ============================================================================

def scale_features(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        scaling_method: str = "standard"
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """
    Scale numeric features using StandardScaler or RobustScaler (PATH B)

    Args:
        X_train: Training features
        X_test: Test features
        scaling_method: "standard" (StandardScaler) or "robust" (RobustScaler)

    Returns:
        Scaled X_train, scaled X_test, scaler objects dict
    """
    log.info("="*80)
    log.info("🔄 PHASE 3.0: FEATURE SCALING (PATH B)")
    log.info("="*80)

    # Get numeric columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    log.info(f"Found {len(numeric_cols)} numeric columns to scale")
    log.info(f"Found {len(categorical_cols)} categorical columns (unchanged)")

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    scalers = {}

    if scaling_method == "robust":
        log.info("Using RobustScaler (handles outliers better)")
        scaler = RobustScaler()
    else:
        log.info("Using StandardScaler (default)")
        scaler = StandardScaler()

    # Fit scaler on train data, transform both train and test
    if len(numeric_cols) > 0:
        X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

        # Store scaler
        scalers['numeric_scaler'] = scaler
        scalers['numeric_cols'] = numeric_cols

        log.info(f"✅ Scaled {len(numeric_cols)} numeric columns")
        log.info(f"   Train shape: {X_train_scaled.shape}")
        log.info(f"   Test shape: {X_test_scaled.shape}")

    log.info("="*80)

    return X_train_scaled, X_test_scaled, scalers


# ============================================================================
# PHASE 3.1: DETECT PROBLEM TYPE FROM TARGET
# ============================================================================

def detect_problem_type_from_target(y_train: pd.Series) -> str:
    """
    Auto-detect classification vs regression from target variable ONLY
    No catalog dependencies
    """
    log.info("="*80)
    log.info("PHASE 3.1: DETECTING PROBLEM TYPE FROM TARGET")
    log.info("="*80)

    # FIX: Handle case where y_train is passed as DataFrame instead of Series
    if isinstance(y_train, pd.DataFrame):
        log.info("ℹ️  Input is DataFrame, converting to Series...")
        y_train = y_train.iloc[:, 0]

    # Check unique values
    n_unique = y_train.nunique()
    unique_ratio = float(n_unique / len(y_train))

    log.info(f"🔍 Target stats: {n_unique} unique values, {unique_ratio:.2%} ratio")

    # Classification: object/bool type or <20 unique values with <10% ratio
    if y_train.dtype in ['object', 'bool', 'category'] or (n_unique < 20 and unique_ratio < 0.1):
        problem_type = 'classification'
        log.info("✅ Detected: CLASSIFICATION")
    else:
        problem_type = 'regression'
        log.info("✅ Detected: REGRESSION")

    return problem_type


# ============================================================================
# PHASE 3.2: TRAIN BASELINE MODEL WITH SCALED FEATURES
# ============================================================================

def train_baseline_model(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        params: Dict[str, Any]
) -> Tuple[object, Dict[str, float], str]:
    """
    Train simple baseline model with scaled features + 5-fold cross-validation
    Returns: model, metrics, problem_type
    """
    log.info("="*80)
    log.info("PHASE 3.2: TRAINING BASELINE MODEL WITH SCALED FEATURES")
    log.info("="*80)

    # FIX: Handle DataFrame input for y_train
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]

    # Detect problem type
    problem_type = detect_problem_type_from_target(y_train)

    if problem_type == 'classification':
        log.info("🎯 Training LogisticRegression baseline...")
        baseline = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        baseline.fit(X_train, y_train)
        train_score = baseline.score(X_train, y_train)
        log.info(f"✅ Baseline train accuracy: {train_score:.4f}")
        metrics = {'accuracy': float(train_score)}

        # ════════════════════════════════════════════════════════════════════
        # ✨ PATH A: 5-FOLD CROSS-VALIDATION
        # ════════════════════════════════════════════════════════════════════

        log.info("="*80)
        log.info("📊 RUNNING 5-FOLD CROSS-VALIDATION (PATH A)")
        log.info("="*80)

        # Use StratifiedKFold to maintain class distribution in each fold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(baseline, X_train, y_train, cv=skf, scoring='accuracy')

        log.info(f"Cross-validation scores for each fold: {[f'{score:.4f}' for score in cv_scores]}")
        log.info(f"Mean CV Score: {cv_scores.mean():.4f}")
        log.info(f"Std Dev: {cv_scores.std():.4f}")
        log.info(f"Confidence Interval: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        log.info("="*80)

        # Store CV scores in model object for later use
        baseline.cv_scores = cv_scores
        baseline.cv_mean = cv_scores.mean()
        baseline.cv_std = cv_scores.std()

        # Update metrics with CV info
        metrics.update({
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'cv_scores': [float(s) for s in cv_scores]
        })

    else:
        log.info("🎯 Training LinearRegression baseline...")
        baseline = LinearRegression()
        baseline.fit(X_train, y_train)
        y_pred = baseline.predict(X_train)
        r2 = r2_score(y_train, y_pred)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        log.info(f"✅ Baseline train R²: {r2:.4f}, RMSE: {rmse:.4f}")
        metrics = {'r2': float(r2), 'rmse': float(rmse)}

    return baseline, metrics, problem_type


# ============================================================================
# PHASE 3.3: ADVANCED HYPERPARAMETER TUNING (PATH B ENHANCED)
# ============================================================================

def hyperparameter_tuning(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        problem_type: str,
        params: Dict[str, Any]
) -> Tuple[object, Dict[str, Any]]:
    """
    Advanced GridSearchCV for hyperparameter tuning with expanded param grids (PATH B)
    """
    log.info("="*80)
    log.info("PHASE 3.3: ADVANCED HYPERPARAMETER TUNING WITH GRIDSEARCHCV (PATH B)")
    log.info("="*80)

    # FIX: Handle DataFrame input for y_train
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]

    if problem_type == 'classification':
        log.info("🎯 Advanced tuning for RandomForestClassifier...")
        model = RandomForestClassifier(random_state=42, n_jobs=-1)

        # PATH B: Expanded parameter grid for better optimization
        param_grid = {
            'n_estimators': [100, 150, 200],      # More options
            'max_depth': [5, 10, 15, 20],         # Deeper trees
            'min_samples_split': [2, 5, 10],      # More options
            'min_samples_leaf': [1, 2, 4],        # New parameter
            'max_features': ['sqrt', 'log2'],     # New parameter
            'bootstrap': [True]
        }

        grid_search = GridSearchCV(
            model, param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)

        log.info(f"✅ Best params: {grid_search.best_params_}")
        log.info(f"✅ Best CV accuracy: {grid_search.best_score_:.4f}")
        log.info(f"✅ Mean test score: {grid_search.cv_results_['mean_test_score'].mean():.4f}")

        tuning_info = {
            'best_params': grid_search.best_params_,
            'best_score': float(grid_search.best_score_),
            'algorithm': 'RandomForestClassifier',
            'grid_size': len(param_grid),
            'cv_folds': 5
        }
    else:
        log.info("🎯 Advanced tuning for GradientBoostingRegressor...")
        model = GradientBoostingRegressor(random_state=42)

        # PATH B: Expanded parameter grid
        param_grid = {
            'n_estimators': [100, 150, 200],
            'learning_rate': [0.001, 0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'subsample': [0.8, 1.0]
        }

        grid_search = GridSearchCV(
            model, param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)

        log.info(f"✅ Best params: {grid_search.best_params_}")
        log.info(f"✅ Best CV R²: {grid_search.best_score_:.4f}")

        tuning_info = {
            'best_params': grid_search.best_params_,
            'best_score': float(grid_search.best_score_),
            'algorithm': 'GradientBoostingRegressor',
            'grid_size': len(param_grid),
            'cv_folds': 5
        }

    log.info("="*80)
    return grid_search.best_estimator_, tuning_info


# ============================================================================
# PHASE 3.4: EVALUATE MODEL
# ============================================================================

def evaluate_model(
        model: object,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        problem_type: str
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation on train and test sets
    """
    log.info("="*80)
    log.info("PHASE 3.4: COMPREHENSIVE MODEL EVALUATION")
    log.info("="*80)

    # FIX: Handle DataFrame input for y_train and y_test
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    if problem_type == 'classification':
        log.info("🎯 Classification metrics:")
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        log.info(f"  ✅ Train accuracy: {train_acc:.4f}")
        log.info(f"  ✅ Test accuracy: {test_acc:.4f}")
        log.info(f"  ✅ Overfit gap: {(train_acc - test_acc):.4f}")

        # ════════════════════════════════════════════════════════════════════
        # ✨ PATH B: ROC-AUC & CONFUSION MATRIX (NEW)
        # ════════════════════════════════════════════════════════════════════

        log.info("-"*80)
        log.info("🎯 PATH B: ROC-AUC & CONFUSION MATRIX ANALYSIS")
        log.info("-"*80)

        # Get probabilities for ROC curve
        if hasattr(model, 'predict_proba'):
            y_test_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_test_proba)
            log.info(f"✅ ROC-AUC Score: {roc_auc:.4f}")
        else:
            roc_auc = None
            log.info("⚠️  Model doesn't support predict_proba")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_test_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0, 0], 0, 0, cm[-1, -1])
        log.info(f"✅ Confusion Matrix:")
        log.info(f"   True Negatives: {tn}, False Positives: {fp}")
        log.info(f"   False Negatives: {fn}, True Positives: {tp}")

        evaluation = {
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'overfit_gap': float(train_acc - test_acc),
            'train_precision': float(precision_score(y_train, y_train_pred, average='weighted', zero_division=0)),
            'test_precision': float(precision_score(y_test, y_test_pred, average='weighted', zero_division=0)),
            'train_f1': float(f1_score(y_train, y_train_pred, average='weighted', zero_division=0)),
            'test_f1': float(f1_score(y_test, y_test_pred, average='weighted', zero_division=0)),
            'roc_auc': float(roc_auc) if roc_auc else None,
            'confusion_matrix': cm.tolist(),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        }

        log.info("\n📋 Classification Report (Test Set):")
        log.info(classification_report(y_test, y_test_pred))

    else:
        log.info("🎯 Regression metrics:")
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        log.info(f"  ✅ Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        log.info(f"  ✅ Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
        log.info(f"  ✅ Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")

        evaluation = {
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'train_rmse': float(train_rmse),
            'test_rmse': float(test_rmse),
            'train_mae': float(train_mae),
            'test_mae': float(test_mae),
            'overfit_gap': float(train_r2 - test_r2),
        }

    return evaluation


# ============================================================================
# PHASE 3.5: SAVE MODEL & EVALUATION
# ============================================================================

def save_model_and_evaluation(
        model: object,
        evaluation: Dict[str, Any],
        problem_type: str
) -> str:
    """
    Save model and evaluation metrics for production
    """
    log.info("="*80)
    log.info("PHASE 3.5: SAVING MODEL & EVALUATION METRICS")
    log.info("="*80)

    import os
    os.makedirs('data/06_models', exist_ok=True)

    # Save model
    model_path = f"data/06_models/best_model_{problem_type}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    log.info(f"✅ Model saved: {model_path}")

    # Save evaluation metrics
    metrics_path = f"data/06_models/model_evaluation_{problem_type}.json"
    with open(metrics_path, 'w') as f:
        json.dump(evaluation, f, indent=2)
    log.info(f"✅ Evaluation saved: {metrics_path}")

    return f"Model saved as {problem_type} model"


# ============================================================================
# PHASE 3.6: MAKE PREDICTIONS
# ============================================================================

def make_predictions(
        model: object,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        problem_type: str
) -> pd.DataFrame:
    """
    Make predictions on test set
    """
    log.info("="*80)
    log.info("PHASE 3.6: MAKING PREDICTIONS")
    log.info("="*80)

    # FIX: Handle DataFrame input for y_test
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]

    predictions = model.predict(X_test)

    if problem_type == 'classification':
        pred_df = pd.DataFrame({
            'actual': y_test.values,
            'predicted': predictions,
            'correct': predictions == y_test.values
        })
        accuracy = (predictions == y_test.values).mean()
        log.info(f"✅ Prediction accuracy: {accuracy:.4f}")
    else:
        residuals = y_test.values - predictions
        pred_df = pd.DataFrame({
            'actual': y_test.values,
            'predicted': predictions,
            'residual': residuals,
            'absolute_error': np.abs(residuals)
        })
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals**2))
        log.info(f"✅ MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    import os
    os.makedirs('data/07_model_output', exist_ok=True)
    pred_path = "data/07_model_output/phase3_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    log.info(f"✅ Predictions saved: {pred_path}")

    return pred_df


# ============================================================================
# PHASE 3: CREATE PIPELINE
# ============================================================================

def create_pipeline(**kwargs) -> Pipeline:
    """
    Complete Phase 3 pipeline: Model Training & Evaluation

    UPDATED WITH PATH B:
    - Added feature scaling (StandardScaler)
    - Enhanced hyperparameter tuning (expanded param grids)
    - Added ROC-AUC & Confusion Matrix analysis
    - Added 5-fold cross-validation

    Inputs (from Phase 2):
      - X_train_selected: Final engineered features
      - X_test_selected: Final engineered features
      - y_train: Training target (Series or DataFrame)
      - y_test: Test target (Series or DataFrame)

    Outputs:
      - X_train_scaled, X_test_scaled, scalers (PATH B)
      - baseline_model, baseline_metrics
      - best_model, tuning_info
      - model_evaluation (with ROC-AUC & confusion matrix)
      - phase3_predictions
    """

    return Pipeline([
        node(
            func=scale_features,
            inputs=["X_train_selected", "X_test_selected"],
            outputs=["X_train_scaled", "X_test_scaled", "scalers"],
            name="phase3_scale_features"
        ),

        node(
            func=train_baseline_model,
            inputs=["X_train_scaled", "y_train", "params:problem_type"],
            outputs=["baseline_model", "baseline_metrics", "problem_type"],
            name="phase3_train_baseline"
        ),

        node(
            func=hyperparameter_tuning,
            inputs=["X_train_scaled", "y_train", "problem_type", "params:feature_selection"],
            outputs=["best_model", "tuning_info"],
            name="phase3_hyperparameter_tuning"
        ),

        node(
            func=evaluate_model,
            inputs=["best_model", "X_train_scaled", "X_test_scaled", "y_train", "y_test", "problem_type"],
            outputs="model_evaluation",
            name="phase3_evaluate_model"
        ),

        node(
            func=save_model_and_evaluation,
            inputs=["best_model", "model_evaluation", "problem_type"],
            outputs="phase3_save_status",
            name="phase3_save_model"
        ),

        node(
            func=make_predictions,
            inputs=["best_model", "X_test_scaled", "y_test", "problem_type"],
            outputs="phase3_predictions",
            name="phase3_make_predictions"
        ),
    ])