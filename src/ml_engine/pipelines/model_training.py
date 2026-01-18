"""
PHASE 3: MODEL TRAINING & EVALUATION (FIXED)
================================================================================
Fixed version that correctly reads problem_type from Phase 2 output
Inputs: X_train_selected, X_test_selected, y_train, y_test, problem_type
Outputs: baseline_model, best_model, model_evaluation, phase3_predictions
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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)
from kedro.pipeline import Pipeline, node

log = logging.getLogger(__name__)


# ============================================================================
# PHASE 3.1: ENSURE PROBLEM TYPE IS STRING
# ============================================================================

def ensure_problem_type_string(problem_type: Any) -> str:
    """
    Ensure problem_type is a clean string value
    Handles if it comes as dict, object, or already a string
    """
    log.info("="*80)
    log.info("PHASE 3.1: ENSURING PROBLEM TYPE IS VALID STRING")
    log.info("="*80)

    # If it's a dict (from problem_type_result), extract the value
    if isinstance(problem_type, dict):
        log.info(f"Problem type is dict: {problem_type}")
        if 'problem_type' in problem_type:
            problem_type = problem_type['problem_type']
        else:
            # Get first value if it's a dict
            problem_type = list(problem_type.values())[0] if problem_type else 'classification'

    # Convert to string and clean
    problem_type = str(problem_type).lower().strip()

    # Validate
    if problem_type not in ['classification', 'regression']:
        log.warning(f"Unknown problem type: {problem_type}, defaulting to classification")
        problem_type = 'classification'

    log.info(f"✅ Problem type: {problem_type}")
    return problem_type


# ============================================================================
# PHASE 3.2: TRAIN BASELINE MODEL
# ============================================================================

def train_baseline_model(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        problem_type: str,
        params: Dict[str, Any]
) -> Tuple[object, Dict[str, float]]:
    """
    Train simple baseline model for comparison
    """
    log.info("="*80)
    log.info("PHASE 3.2: TRAINING BASELINE MODEL")
    log.info("="*80)

    # Ensure problem_type is string
    problem_type = ensure_problem_type_string(problem_type)

    if problem_type == 'classification':
        log.info("🎯 Training LogisticRegression baseline...")
        baseline = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        baseline.fit(X_train, y_train)
        train_score = baseline.score(X_train, y_train)
        log.info(f"✅ Baseline train accuracy: {train_score:.4f}")
        metrics = {'accuracy': float(train_score)}
    else:
        log.info("🎯 Training LinearRegression baseline...")
        baseline = LinearRegression()
        baseline.fit(X_train, y_train)
        y_pred = baseline.predict(X_train)
        r2 = r2_score(y_train, y_pred)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        log.info(f"✅ Baseline train R²: {r2:.4f}, RMSE: {rmse:.4f}")
        metrics = {'r2': float(r2), 'rmse': float(rmse)}

    return baseline, metrics


# ============================================================================
# PHASE 3.3: HYPERPARAMETER TUNING
# ============================================================================

def hyperparameter_tuning(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        problem_type: str,
        params: Dict[str, Any]
) -> Tuple[object, Dict[str, Any]]:
    """
    GridSearchCV for hyperparameter tuning with cross-validation
    """
    log.info("="*80)
    log.info("PHASE 3.3: HYPERPARAMETER TUNING WITH GRIDSEARCHCV")
    log.info("="*80)

    # Ensure problem_type is string
    problem_type = ensure_problem_type_string(problem_type)

    cv_folds = 5

    if problem_type == 'classification':
        log.info("🎯 Tuning RandomForestClassifier...")
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5]
        }

        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        log.info(f"✅ Best params: {grid_search.best_params_}")
        log.info(f"✅ Best CV accuracy: {grid_search.best_score_:.4f}")

        tuning_info = {
            'best_params': grid_search.best_params_,
            'best_score': float(grid_search.best_score_),
            'algorithm': 'RandomForestClassifier'
        }
    else:
        log.info("🎯 Tuning GradientBoostingRegressor...")
        model = GradientBoostingRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5, 7]
        }

        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        log.info(f"✅ Best params: {grid_search.best_params_}")
        log.info(f"✅ Best CV R²: {grid_search.best_score_:.4f}")

        tuning_info = {
            'best_params': grid_search.best_params_,
            'best_score': float(grid_search.best_score_),
            'algorithm': 'GradientBoostingRegressor'
        }

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

    # Ensure problem_type is string
    problem_type = ensure_problem_type_string(problem_type)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    if problem_type == 'classification':
        log.info("🎯 Classification metrics:")
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        log.info(f"  ✅ Train accuracy: {train_acc:.4f}")
        log.info(f"  ✅ Test accuracy: {test_acc:.4f}")
        log.info(f"  ✅ Overfit gap: {(train_acc - test_acc):.4f}")

        evaluation = {
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'overfit_gap': float(train_acc - test_acc),
            'train_precision': float(precision_score(y_train, y_train_pred, average='weighted', zero_division=0)),
            'test_precision': float(precision_score(y_test, y_test_pred, average='weighted', zero_division=0)),
            'train_f1': float(f1_score(y_train, y_train_pred, average='weighted', zero_division=0)),
            'test_f1': float(f1_score(y_test, y_test_pred, average='weighted', zero_division=0)),
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

    # Ensure problem_type is string
    problem_type = ensure_problem_type_string(problem_type)

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

    # Ensure problem_type is string
    problem_type = ensure_problem_type_string(problem_type)

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

    Inputs (from Phase 2):
      - X_train_selected: Final engineered features
      - X_test_selected: Final engineered features
      - y_train: Training target
      - y_test: Test target
      - problem_type: 'classification' or 'regression' (from Phase 2)

    Outputs:
      - baseline_model, baseline_metrics
      - best_model, tuning_info
      - model_evaluation
      - phase3_predictions
    """

    return Pipeline([
        node(
            func=train_baseline_model,
            inputs=["X_train_selected", "y_train", "problem_type", "params:problem_type"],
            outputs=["baseline_model", "baseline_metrics"],
            name="phase3_train_baseline"
        ),

        node(
            func=hyperparameter_tuning,
            inputs=["X_train_selected", "y_train", "problem_type", "params:feature_selection"],
            outputs=["best_model", "tuning_info"],
            name="phase3_hyperparameter_tuning"
        ),

        node(
            func=evaluate_model,
            inputs=["best_model", "X_train_selected", "X_test_selected", "y_train", "y_test", "problem_type"],
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
            inputs=["best_model", "X_test_selected", "y_test", "problem_type"],
            outputs="phase3_predictions",
            name="phase3_make_predictions"
        ),
    ])