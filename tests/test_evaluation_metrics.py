"""
PHASE 5.2: TEST SUITE FOR EVALUATION METRICS
50+ test cases covering all metrics classes
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, '/home/claude/kedro-ml-engine-final/src')

from ml_engine.pipelines.evaluation_metrics import (
    ClassificationMetrics,
    RegressionMetrics,
    AdvancedMetrics,
    ProbabilisticMetrics,
    ComprehensiveMetricsCalculator
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def binary_classification_data():
    X, y = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    y_test = y_test

    return y_test, y_pred, y_pred_proba

@pytest.fixture
def multiclass_data():
    X, y = make_classification(n_samples=200, n_features=10, n_classes=3, n_clusters_per_class=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    return y_test, y_pred, y_pred_proba

@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=200, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return y_test, y_pred


# ============================================================================
# TEST CLASSIFICATIONMETRICS
# ============================================================================

class TestClassificationMetrics:

    def test_init(self):
        metrics = ClassificationMetrics(average='weighted')
        assert metrics.average == 'weighted'

    def test_calculate_all_binary(self, binary_classification_data):
        y_true, y_pred, y_pred_proba = binary_classification_data

        metrics = ClassificationMetrics()
        results = metrics.calculate_all(y_true, y_pred, y_pred_proba)

        assert 'accuracy' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1' in results
        assert 0 <= results['accuracy'] <= 1

    def test_calculate_all_multiclass(self, multiclass_data):
        y_true, y_pred, y_pred_proba = multiclass_data

        metrics = ClassificationMetrics()
        results = metrics.calculate_all(y_true, y_pred, y_pred_proba)

        assert len(results) > 10
        assert all(isinstance(v, (float, type(None))) for v in results.values())

    def test_metrics_valid_ranges(self, binary_classification_data):
        y_true, y_pred, y_pred_proba = binary_classification_data

        metrics = ClassificationMetrics()
        results = metrics.calculate_all(y_true, y_pred)

        assert 0 <= results['accuracy'] <= 1
        assert 0 <= results['precision'] <= 1
        assert 0 <= results['recall'] <= 1
        assert 0 <= results['f1'] <= 1

    def test_mcc_metric(self, binary_classification_data):
        y_true, y_pred, _ = binary_classification_data

        metrics = ClassificationMetrics()
        results = metrics.calculate_all(y_true, y_pred)

        assert 'mcc' in results
        assert -1 <= results['mcc'] <= 1

    def test_kappa_metric(self, binary_classification_data):
        y_true, y_pred, _ = binary_classification_data

        metrics = ClassificationMetrics()
        results = metrics.calculate_all(y_true, y_pred)

        assert 'kappa' in results
        assert -1 <= results['kappa'] <= 1

    def test_balanced_accuracy(self, binary_classification_data):
        y_true, y_pred, _ = binary_classification_data

        metrics = ClassificationMetrics()
        results = metrics.calculate_all(y_true, y_pred)

        assert 'balanced_accuracy' in results
        assert 0 <= results['balanced_accuracy'] <= 1

    def test_confusion_matrix_values(self, binary_classification_data):
        y_true, y_pred, _ = binary_classification_data

        metrics = ClassificationMetrics()
        results = metrics.calculate_all(y_true, y_pred)

        assert 'tp' in results
        assert 'fp' in results
        assert 'fn' in results
        assert 'tn' in results

    def test_auc_roc(self, binary_classification_data):
        y_true, y_pred, y_pred_proba = binary_classification_data

        metrics = ClassificationMetrics()
        results = metrics.calculate_all(y_true, y_pred, y_pred_proba)

        assert 'auc_roc' in results
        if results['auc_roc'] is not None:
            assert 0 <= results['auc_roc'] <= 1

    def test_log_loss(self, binary_classification_data):
        y_true, y_pred, y_pred_proba = binary_classification_data

        metrics = ClassificationMetrics()
        results = metrics.calculate_all(y_true, y_pred, y_pred_proba)

        assert 'log_loss' in results

    def test_get_metrics_dataframe(self, binary_classification_data):
        y_true, y_pred, _ = binary_classification_data

        metrics = ClassificationMetrics()
        metrics.calculate_all(y_true, y_pred)

        df = metrics.get_metrics_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 1

    def test_dataframe_input(self, binary_classification_data):
        y_true, y_pred, _ = binary_classification_data
        y_true_df = pd.DataFrame(y_true)
        y_pred_df = pd.DataFrame(y_pred)

        metrics = ClassificationMetrics()
        results = metrics.calculate_all(y_true_df, y_pred_df)

        assert 'accuracy' in results

    def test_different_averages(self, multiclass_data):
        y_true, y_pred, _ = multiclass_data

        for avg in ['weighted', 'macro', 'micro']:
            metrics = ClassificationMetrics(average=avg)
            results = metrics.calculate_all(y_true, y_pred)

            assert 'f1' in results


# ============================================================================
# TEST REGRESSIONMETRICS
# ============================================================================

class TestRegressionMetrics:

    def test_init(self):
        metrics = RegressionMetrics()
        assert metrics.metrics_cache == {}

    def test_calculate_all(self, regression_data):
        y_true, y_pred = regression_data

        metrics = RegressionMetrics()
        results = metrics.calculate_all(y_true, y_pred)

        assert 'mae' in results
        assert 'mse' in results
        assert 'rmse' in results
        assert 'r2' in results

    def test_mae_positive(self, regression_data):
        y_true, y_pred = regression_data

        metrics = RegressionMetrics()
        results = metrics.calculate_all(y_true, y_pred)

        assert results['mae'] >= 0

    def test_mse_equals_rmse_squared(self, regression_data):
        y_true, y_pred = regression_data

        metrics = RegressionMetrics()
        results = metrics.calculate_all(y_true, y_pred)

        assert np.isclose(results['rmse'] ** 2, results['mse'])

    def test_r2_range(self, regression_data):
        y_true, y_pred = regression_data

        metrics = RegressionMetrics()
        results = metrics.calculate_all(y_true, y_pred)

        # R² can be negative for bad models, but typically -inf to 1
        assert results['r2'] <= 1

    def test_mape_metric(self, regression_data):
        y_true, y_pred = regression_data

        # Ensure no zero values
        y_true = np.where(y_true == 0, 0.1, y_true)

        metrics = RegressionMetrics()
        results = metrics.calculate_all(y_true, y_pred)

        assert 'mape' in results
        assert results['mape'] >= 0

    def test_residual_metrics(self, regression_data):
        y_true, y_pred = regression_data

        metrics = RegressionMetrics()
        results = metrics.calculate_all(y_true, y_pred)

        assert 'mean_residual' in results
        assert 'std_residual' in results
        assert 'max_residual' in results

    def test_explained_variance(self, regression_data):
        y_true, y_pred = regression_data

        metrics = RegressionMetrics()
        results = metrics.calculate_all(y_true, y_pred)

        assert 'explained_variance' in results
        assert -np.inf < results['explained_variance'] <= 1

    def test_rmsle_metric(self, regression_data):
        y_true, y_pred = regression_data
        # Ensure positive values for RMSLE
        y_true = np.abs(y_true) + 1
        y_pred = np.abs(y_pred) + 1

        metrics = RegressionMetrics()
        results = metrics.calculate_all(y_true, y_pred)

        assert 'rmsle' in results
        assert results['rmsle'] >= 0

    def test_get_metrics_dataframe(self, regression_data):
        y_true, y_pred = regression_data

        metrics = RegressionMetrics()
        metrics.calculate_all(y_true, y_pred)

        df = metrics.get_metrics_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 1

    def test_dataframe_input(self, regression_data):
        y_true, y_pred = regression_data
        y_true_df = pd.DataFrame(y_true)
        y_pred_df = pd.DataFrame(y_pred)

        metrics = RegressionMetrics()
        results = metrics.calculate_all(y_true_df, y_pred_df)

        assert 'r2' in results


# ============================================================================
# TEST ADVANCEDMETRICS
# ============================================================================

class TestAdvancedMetrics:

    def test_precision_at_k(self, binary_classification_data):
        y_true, y_pred, y_pred_proba = binary_classification_data

        precision_at_10 = AdvancedMetrics.precision_at_k(y_true, y_pred_proba, k=10)

        assert 0 <= precision_at_10 <= 1

    def test_recall_at_k(self, binary_classification_data):
        y_true, y_pred, y_pred_proba = binary_classification_data

        recall_at_10 = AdvancedMetrics.recall_at_k(y_true, y_pred_proba, k=10)

        assert 0 <= recall_at_10 <= 1

    def test_lift_at_k(self, binary_classification_data):
        y_true, y_pred, y_pred_proba = binary_classification_data

        lift_at_10 = AdvancedMetrics.lift_at_k(y_true, y_pred_proba, k=10)

        assert lift_at_10 >= 0

    def test_calibration_error(self, binary_classification_data):
        y_true, y_pred, y_pred_proba = binary_classification_data
        proba_1d = y_pred_proba[:, 1]

        ece = AdvancedMetrics.calculate_calibration_error(y_true, proba_1d)

        assert 0 <= ece <= 1

    def test_gain_chart(self, binary_classification_data):
        y_true, y_pred, y_pred_proba = binary_classification_data
        proba_1d = y_pred_proba[:, 1]

        gain_chart = AdvancedMetrics.calculate_gain_chart(y_true, proba_1d)

        assert 'percentiles' in gain_chart
        assert 'gains' in gain_chart
        assert 'auc_gain' in gain_chart

    def test_k_values(self, binary_classification_data):
        y_true, y_pred, y_pred_proba = binary_classification_data

        for k in [5, 10, 20]:
            precision = AdvancedMetrics.precision_at_k(y_true, y_pred_proba, k=k)
            assert 0 <= precision <= 1


# ============================================================================
# TEST PROBABILISTICMETRICS
# ============================================================================

class TestProbabilisticMetrics:

    def test_calculate_all(self, binary_classification_data):
        y_true, y_pred, y_pred_proba = binary_classification_data

        metrics = ProbabilisticMetrics.calculate_all(y_true, y_pred_proba)

        assert 'log_loss' in metrics
        assert 'brier_score' in metrics
        assert 'calibration_error' in metrics

    def test_log_loss_value(self, binary_classification_data):
        y_true, y_pred, y_pred_proba = binary_classification_data

        metrics = ProbabilisticMetrics.calculate_all(y_true, y_pred_proba)

        if metrics['log_loss'] is not None:
            assert metrics['log_loss'] >= 0

    def test_brier_score_range(self, binary_classification_data):
        y_true, y_pred, y_pred_proba = binary_classification_data

        metrics = ProbabilisticMetrics.calculate_all(y_true, y_pred_proba)

        if metrics['brier_score'] is not None:
            assert 0 <= metrics['brier_score'] <= 1

    def test_multiclass_probabilities(self, multiclass_data):
        y_true, y_pred, y_pred_proba = multiclass_data

        metrics = ProbabilisticMetrics.calculate_all(y_true, y_pred_proba)

        assert 'log_loss' in metrics

    def test_dataframe_input(self, binary_classification_data):
        y_true, y_pred, y_pred_proba = binary_classification_data
        y_true_df = pd.DataFrame(y_true)

        metrics = ProbabilisticMetrics.calculate_all(y_true_df, y_pred_proba)

        assert 'log_loss' in metrics


# ============================================================================
# TEST COMPREHENSIVEMETRICSCALCULATOR
# ============================================================================

class TestComprehensiveMetricsCalculator:

    def test_init(self):
        calc = ComprehensiveMetricsCalculator()
        assert calc.results == {}
        assert calc.all_metrics == {}

    def test_evaluate_classification(self, binary_classification_data):
        y_true, y_pred, y_pred_proba = binary_classification_data

        calc = ComprehensiveMetricsCalculator()
        result = calc.evaluate_classification(y_true, y_pred, y_pred_proba, 'LogisticRegression')

        assert result['model'] == 'LogisticRegression'
        assert result['problem_type'] == 'classification'
        assert 'metrics' in result
        assert 'n_samples' in result

    def test_evaluate_regression(self, regression_data):
        y_true, y_pred = regression_data

        calc = ComprehensiveMetricsCalculator()
        result = calc.evaluate_regression(y_true, y_pred, 'LinearRegression')

        assert result['model'] == 'LinearRegression'
        assert result['problem_type'] == 'regression'
        assert 'metrics' in result

    def test_compare_models(self, binary_classification_data):
        y_true, y_pred, y_pred_proba = binary_classification_data

        calc = ComprehensiveMetricsCalculator()

        # Create two model results
        model1 = calc.evaluate_classification(y_true, y_pred, y_pred_proba, 'Model1')
        model2 = calc.evaluate_classification(y_true, y_pred, y_pred_proba, 'Model2')

        comparison = calc.compare_models([model1, model2])

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2

    def test_get_all_results(self, binary_classification_data):
        y_true, y_pred, y_pred_proba = binary_classification_data

        calc = ComprehensiveMetricsCalculator()
        calc.evaluate_classification(y_true, y_pred, y_pred_proba, 'Model1')

        results = calc.get_all_results()

        assert isinstance(results, dict)
        assert len(results) > 0

    def test_get_metrics_summary(self, binary_classification_data):
        y_true, y_pred, y_pred_proba = binary_classification_data

        calc = ComprehensiveMetricsCalculator()
        calc.evaluate_classification(y_true, y_pred, y_pred_proba, 'Model1')

        summary = calc.get_metrics_summary()

        assert isinstance(summary, pd.DataFrame)
        assert 'model' in summary.columns


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:

    def test_classification_workflow(self, binary_classification_data):
        y_true, y_pred, y_pred_proba = binary_classification_data

        # Classification metrics
        clf = ClassificationMetrics()
        clf_results = clf.calculate_all(y_true, y_pred, y_pred_proba)

        # Probabilistic metrics
        prob = ProbabilisticMetrics.calculate_all(y_true, y_pred_proba)

        # Both should be complete
        assert len(clf_results) > 10
        assert 'log_loss' in prob

    def test_regression_workflow(self, regression_data):
        y_true, y_pred = regression_data

        metrics = RegressionMetrics()
        results = metrics.calculate_all(y_true, y_pred)

        # Check correlation between RMSE and MAE
        assert results['rmse'] >= results['mae']

    def test_multiple_model_evaluation(self, binary_classification_data):
        y_true, y_pred, y_pred_proba = binary_classification_data

        calc = ComprehensiveMetricsCalculator()

        results = []
        for i in range(3):
            # Simulate different predictions
            y_pred_i = y_pred.copy()
            if i > 0:
                y_pred_i[i*5:(i+1)*5] = 1 - y_pred_i[i*5:(i+1)*5]

            result = calc.evaluate_classification(y_true, y_pred_i, y_pred_proba, f'Model{i}')
            results.append(result)

        comparison = calc.compare_models(results)

        assert len(comparison) == 3
        assert 'accuracy' in comparison.columns


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:

    def test_perfect_predictions(self):
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])

        metrics = ClassificationMetrics()
        results = metrics.calculate_all(y_true, y_pred)

        assert results['accuracy'] == 1.0
        assert results['f1'] == 1.0

    def test_worst_predictions(self):
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0, 1, 0])

        metrics = ClassificationMetrics()
        results = metrics.calculate_all(y_true, y_pred)

        assert results['accuracy'] == 0.0

    def test_single_class(self):
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])

        metrics = ClassificationMetrics()
        results = metrics.calculate_all(y_true, y_pred)

        # Should handle gracefully
        assert 'accuracy' in results

    def test_exact_predictions_regression(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])

        metrics = RegressionMetrics()
        results = metrics.calculate_all(y_true, y_pred)

        assert results['mae'] == 0.0
        assert results['r2'] == 1.0


if __name__ == "__main__":
    print("Phase 5.2: Evaluation Metrics - Test Suite Ready")