"""
PHASE 4: COMPLETE ML ALGORITHMS (FINAL FIX)
================================================================================
FIXED: Removed problem_type_result[problem_type] reference
Uses simple "problem_type" string input
================================================================================
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, List
import pickle
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    LogisticRegression, RidgeClassifier, SGDClassifier,
    PassiveAggressiveRegressor, PassiveAggressiveClassifier,
    HuberRegressor, RANSACRegressor, TheilSenRegressor,
    BayesianRidge, ARDRegression, Lars, LassoLars, Perceptron
)
from sklearn.svm import SVR, SVC, LinearSVR, LinearSVC, NuSVR, NuSVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    AdaBoostRegressor, AdaBoostClassifier,
    BaggingRegressor, BaggingClassifier
)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from kedro.pipeline import Pipeline, node

try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

log = logging.getLogger(__name__)


# ============================================================================
# PHASE 4.1: GET ALL REGRESSION ALGORITHMS (24+)
# ============================================================================

def get_regression_algorithms() -> Dict[str, object]:
    """Get all regression algorithms"""
    log.info("Loading regression algorithms...")

    algorithms = {
        # Linear Models (8)
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'BayesianRidge': BayesianRidge(),
        'ARDRegression': ARDRegression(),
        'HuberRegressor': HuberRegressor(),
        'Lars': Lars(),

        # Tree-based (6)
        'DecisionTreeRegressor': DecisionTreeRegressor(max_depth=10, random_state=42),
        'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'ExtraTreesRegressor': ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'GradientBoostingRegressor': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'AdaBoostRegressor': AdaBoostRegressor(n_estimators=100, random_state=42),
        'BaggingRegressor': BaggingRegressor(n_estimators=100, random_state=42, n_jobs=-1),

        # Support Vector Machines (3)
        'SVR': SVR(kernel='rbf'),
        'LinearSVR': LinearSVR(random_state=42),
        'NuSVR': NuSVR(kernel='rbf'),

        # Specialized (3)
        'RANSACRegressor': RANSACRegressor(random_state=42),
        'TheilSenRegressor': TheilSenRegressor(random_state=42),
        'PassiveAggressiveRegressor': PassiveAggressiveRegressor(random_state=42),
    }

    # Add optional algorithms
    if XGBOOST_AVAILABLE:
        algorithms['XGBRegressor'] = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0)
        log.info("✅ XGBoost available")

    if LIGHTGBM_AVAILABLE:
        algorithms['LGBMRegressor'] = LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
        log.info("✅ LightGBM available")

    if CATBOOST_AVAILABLE:
        algorithms['CatBoostRegressor'] = CatBoostRegressor(iterations=100, random_state=42, verbose=False)
        log.info("✅ CatBoost available")

    log.info(f"✅ Loaded {len(algorithms)} regression algorithms")
    return algorithms


# ============================================================================
# PHASE 4.2: GET ALL CLASSIFICATION ALGORITHMS (26+)
# ============================================================================

def get_classification_algorithms() -> Dict[str, object]:
    """Get all classification algorithms"""
    log.info("Loading classification algorithms...")

    algorithms = {
        # Linear Models (5)
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        'RidgeClassifier': RidgeClassifier(alpha=1.0),
        'SGDClassifier': SGDClassifier(max_iter=1000, random_state=42, n_jobs=-1),
        'Perceptron': Perceptron(random_state=42, n_jobs=-1),
        'PassiveAggressiveClassifier': PassiveAggressiveClassifier(random_state=42, n_jobs=-1),

        # Tree-based (6)
        'DecisionTreeClassifier': DecisionTreeClassifier(max_depth=10, random_state=42),
        'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'ExtraTreesClassifier': ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'AdaBoostClassifier': AdaBoostClassifier(n_estimators=100, random_state=42),
        'BaggingClassifier': BaggingClassifier(n_estimators=100, random_state=42, n_jobs=-1),

        # Support Vector Machines (4)
        'SVC_RBF': SVC(kernel='rbf', random_state=42, probability=True),
        'LinearSVC': LinearSVC(max_iter=2000, random_state=42, dual=False),
        'NuSVC': NuSVC(kernel='rbf', random_state=42, probability=True),
        'SVC_Poly': SVC(kernel='poly', degree=2, random_state=42, probability=True),

        # Naive Bayes (5)
        'GaussianNB': GaussianNB(),
        'MultinomialNB': MultinomialNB(),
        'BernoulliNB': BernoulliNB(),
        'ComplementNB': ComplementNB(),
        'CategoricalNB': CategoricalNB(),

        # Neighbors (1)
        'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }

    # Add optional algorithms
    if XGBOOST_AVAILABLE:
        algorithms['XGBClassifier'] = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0)
        log.info("✅ XGBoost available")

    if LIGHTGBM_AVAILABLE:
        algorithms['LGBMClassifier'] = LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
        log.info("✅ LightGBM available")

    if CATBOOST_AVAILABLE:
        algorithms['CatBoostClassifier'] = CatBoostClassifier(iterations=100, random_state=42, verbose=False)
        log.info("✅ CatBoost available")

    log.info(f"✅ Loaded {len(algorithms)} classification algorithms")
    return algorithms


# ============================================================================
# PHASE 4.3: TRAIN ALL ALGORITHMS
# ============================================================================

def phase4_train_all_algorithms(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        problem_type: str,
        params: Dict[str, Any]
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """Train all algorithms and return results"""

    log.info("="*80)
    log.info("PHASE 4: TRAINING ALL ALGORITHMS")
    log.info("="*80)

    # Ensure problem_type is a simple string
    problem_type = str(problem_type).lower().strip() if problem_type else 'classification'
    log.info(f"Problem type: {problem_type}")

    if problem_type == 'classification':
        algorithms = get_classification_algorithms()
    else:
        algorithms = get_regression_algorithms()

    trained_models = {}
    results = []

    for algo_name, model in algorithms.items():
        try:
            log.info(f"Training {algo_name}...")

            # Handle categorical features for Naive Bayes
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()

            if 'NB' in algo_name:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
                X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
                X_train_scaled = X_train_scaled - X_train_scaled.min() + 1e-10
                X_test_scaled = X_test_scaled - X_test_scaled.min() + 1e-10

            # Train
            model.fit(X_train_scaled, y_train)
            trained_models[algo_name] = model

            # Evaluate
            if problem_type == 'classification':
                train_pred = model.predict(X_train_scaled)
                test_pred = model.predict(X_test_scaled)
                train_score = accuracy_score(y_train, train_pred)
                test_score = accuracy_score(y_test, test_pred)
                metric = 'accuracy'
            else:
                train_pred = model.predict(X_train_scaled)
                test_pred = model.predict(X_test_scaled)
                train_score = r2_score(y_train, train_pred)
                test_score = r2_score(y_test, test_pred)
                metric = 'r2'

            gap = train_score - test_score
            results.append({
                'Algorithm': algo_name,
                'Train_Score': train_score,
                'Test_Score': test_score,
                'Overfit_Gap': gap,
                'Metric': metric
            })

            log.info(f"  ✅ {algo_name}: Train={train_score:.4f}, Test={test_score:.4f}")

        except Exception as e:
            log.warning(f"  ❌ {algo_name} failed: {str(e)}")
            continue

    results_df = pd.DataFrame(results).sort_values('Test_Score', ascending=False)

    return trained_models, results_df


# ============================================================================
# PHASE 4.4: GENERATE COMPARISON REPORT
# ============================================================================

def phase4_generate_report(
        trained_models: Dict[str, object],
        results_df: pd.DataFrame,
        problem_type: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generate comprehensive comparison report"""

    log.info("="*80)
    log.info("PHASE 4: ALGORITHM COMPARISON REPORT")
    log.info("="*80)

    report = results_df.copy()
    report['Rank'] = range(1, len(report) + 1)

    log.info("\n📊 TOP 10 ALGORITHMS:")
    log.info(report.head(10).to_string())

    # Select top 5
    top_5 = report.head(5)['Algorithm'].tolist()

    # Summary statistics
    summary = {
        'total_algorithms': len(trained_models),
        'top_algorithm': report.iloc[0]['Algorithm'],
        'top_score': float(report.iloc[0]['Test_Score']),
        'top_5_algorithms': top_5,
        'avg_test_score': float(report['Test_Score'].mean()),
        'best_test_score': float(report['Test_Score'].max()),
        'worst_test_score': float(report['Test_Score'].min()),
    }

    log.info(f"\n✅ Total algorithms trained: {summary['total_algorithms']}")
    log.info(f"✅ Best algorithm: {summary['top_algorithm']} ({summary['top_score']:.4f})")
    log.info(f"✅ Average test score: {summary['avg_test_score']:.4f}")

    return report, summary


# ============================================================================
# PHASE 4.5: SAVE ALL RESULTS
# ============================================================================

def phase4_save_results(
        trained_models: Dict[str, object],
        results_df: pd.DataFrame,
        report: pd.DataFrame,
        summary: Dict[str, Any],
        problem_type: str
) -> str:
    """Save all models and results"""

    log.info("="*80)
    log.info("PHASE 4: SAVING ALL RESULTS")
    log.info("="*80)

    os.makedirs('data/06_models/phase4', exist_ok=True)
    os.makedirs('data/07_model_output', exist_ok=True)

    # Save comparison results
    csv_path = f'data/07_model_output/phase4_algorithm_comparison.csv'
    results_df.to_csv(csv_path, index=False)
    log.info(f"✅ Saved comparison: {csv_path}")

    # Save report
    report_path = f'data/07_model_output/phase4_report.csv'
    report.to_csv(report_path, index=False)
    log.info(f"✅ Saved report: {report_path}")

    # Save summary
    summary_path = f'data/07_model_output/phase4_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    log.info(f"✅ Saved summary: {summary_path}")

    # Save best models
    top_5_names = report.head(5)['Algorithm'].tolist()
    best_models = {name: trained_models[name] for name in top_5_names if name in trained_models}

    models_path = f'data/06_models/phase4/best_models_{problem_type}.pkl'
    with open(models_path, 'wb') as f:
        pickle.dump(best_models, f)
    log.info(f"✅ Saved best 5 models: {models_path}")

    # Save all models
    all_models_path = f'data/06_models/phase4/all_trained_models_{problem_type}.pkl'
    with open(all_models_path, 'wb') as f:
        pickle.dump(trained_models, f)
    log.info(f"✅ Saved all {len(trained_models)} models: {all_models_path}")

    return f"Saved {len(trained_models)} models successfully"


# ============================================================================
# PHASE 4: CREATE PIPELINE
# ============================================================================

def create_pipeline(**kwargs) -> Pipeline:
    """
    Complete Phase 4 pipeline: 50+ ML Algorithms

    FINAL FIX: Uses simple "problem_type" string input (not problem_type_result[problem_type])

    Inputs (from Phase 2):
      - X_train_selected: Final engineered features
      - X_test_selected: Final engineered features
      - y_train: Training target
      - y_test: Test target
      - problem_type: Simple string (from Phase 3)

    Outputs:
      - phase4_trained_models: All trained models
      - phase4_results: Results dataframe
      - phase4_report: Ranked report
      - phase4_summary: Summary statistics
      - phase4_save_status: Save confirmation
    """

    return Pipeline([
        node(
            func=phase4_train_all_algorithms,
            inputs=["X_train_selected", "X_test_selected", "y_train", "y_test", "problem_type", "params:problem_type"],
            outputs=["phase4_trained_models", "phase4_results"],
            name="phase4_train_all"
        ),

        node(
            func=phase4_generate_report,
            inputs=["phase4_trained_models", "phase4_results", "problem_type"],
            outputs=["phase4_report", "phase4_summary"],
            name="phase4_generate_report"
        ),

        node(
            func=phase4_save_results,
            inputs=["phase4_trained_models", "phase4_results", "phase4_report", "phase4_summary", "problem_type"],
            outputs="phase4_save_status",
            name="phase4_save_results"
        ),
    ])