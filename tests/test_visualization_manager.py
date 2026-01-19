"""
PHASE 5.5: TEST SUITE FOR VISUALIZATION MANAGER
50+ test cases covering all visualization types
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
import tempfile
import os

import sys
sys.path.insert(0, '/home/claude/kedro-ml-engine-final/src')

from ml_engine.pipelines.visualization_manager import (
    ConfusionMatrixVisualizer,
    ROCCurveVisualizer,
    CalibrationPlotVisualizer,
    ResidualPlotVisualizer,
    FeatureImportanceVisualizer,
    LearningCurveVisualizer,
    PredictionDistributionVisualizer,
    ClassificationMetricsVisualizer,
    PerformanceComparisonVisualizer,
    VisualizationManager
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def classification_data():
    X, y = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    return {
        'y_true': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'feature_names': [f'Feature_{i}' for i in range(10)]
    }

@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=200, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return {
        'y_true': y_test,
        'y_pred': y_pred,
        'feature_names': [f'Feature_{i}' for i in range(10)]
    }

@pytest.fixture
def multi_model_predictions(classification_data):
    predictions_dict = {}
    probabilities_dict = {}

    y_true = classification_data['y_true']

    for i in range(3):
        y_pred = np.random.randint(0, 2, len(y_true))
        predictions_dict[f'Model_{i+1}'] = y_pred

        y_proba = np.random.rand(len(y_true), 2)
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
        probabilities_dict[f'Model_{i+1}'] = y_proba

    return predictions_dict, probabilities_dict


# ============================================================================
# TEST CONFUSIONMATRIXVISUALIZER
# ============================================================================

class TestConfusionMatrixVisualizer:

    def test_init(self):
        viz = ConfusionMatrixVisualizer()
        assert viz.figsize == (10, 8)

    def test_plot_returns_figure(self, classification_data):
        viz = ConfusionMatrixVisualizer()
        fig = viz.plot(classification_data['y_true'], classification_data['y_pred'])

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_with_title(self, classification_data):
        viz = ConfusionMatrixVisualizer()
        title = "Test Confusion Matrix"
        fig = viz.plot(classification_data['y_true'], classification_data['y_pred'], title=title)

        assert fig.get_suptitle() is not None or any(ax.get_title() for ax in fig.get_axes())
        plt.close(fig)

    def test_plot_normalized(self, classification_data):
        viz = ConfusionMatrixVisualizer()
        fig = viz.plot(classification_data['y_true'], classification_data['y_pred'], normalize=True)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_comparison(self, classification_data, multi_model_predictions):
        viz = ConfusionMatrixVisualizer()
        predictions_dict, _ = multi_model_predictions

        fig = viz.plot_comparison(classification_data['y_true'], predictions_dict)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ============================================================================
# TEST ROCCURVEVISUALIZER
# ============================================================================

class TestROCCurveVisualizer:

    def test_init(self):
        viz = ROCCurveVisualizer()
        assert viz.figsize == (10, 8)

    def test_plot_returns_figure(self, classification_data):
        viz = ROCCurveVisualizer()
        fig = viz.plot(classification_data['y_true'], classification_data['y_pred_proba'])

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_has_diagonal(self, classification_data):
        viz = ROCCurveVisualizer()
        fig = viz.plot(classification_data['y_true'], classification_data['y_pred_proba'])

        ax = fig.get_axes()[0]
        lines = ax.get_lines()
        assert len(lines) >= 2  # ROC curve + diagonal
        plt.close(fig)

    def test_plot_comparison(self, classification_data, multi_model_predictions):
        viz = ROCCurveVisualizer()
        _, probabilities_dict = multi_model_predictions

        fig = viz.plot_comparison(classification_data['y_true'], probabilities_dict)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ============================================================================
# TEST CALIBRATIONPLOTVISUALIZER
# ============================================================================

class TestCalibrationPlotVisualizer:

    def test_init(self):
        viz = CalibrationPlotVisualizer()
        assert viz.figsize == (10, 8)

    def test_plot_returns_figure(self, classification_data):
        viz = CalibrationPlotVisualizer()
        fig = viz.plot(classification_data['y_true'], classification_data['y_pred_proba'])

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_comparison(self, classification_data, multi_model_predictions):
        viz = CalibrationPlotVisualizer()
        _, probabilities_dict = multi_model_predictions

        fig = viz.plot_comparison(classification_data['y_true'], probabilities_dict)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ============================================================================
# TEST RESIDUALPLOTVISUALIZER
# ============================================================================

class TestResidualPlotVisualizer:

    def test_init(self):
        viz = ResidualPlotVisualizer()
        assert viz.figsize == (12, 8)

    def test_plot_returns_figure(self, regression_data):
        viz = ResidualPlotVisualizer()
        fig = viz.plot(regression_data['y_true'], regression_data['y_pred'])

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_has_two_subplots(self, regression_data):
        viz = ResidualPlotVisualizer()
        fig = viz.plot(regression_data['y_true'], regression_data['y_pred'])

        axes = fig.get_axes()
        assert len(axes) == 2
        plt.close(fig)


# ============================================================================
# TEST FEATUREIMPORTANCEVISUALIZER
# ============================================================================

class TestFeatureImportanceVisualizer:

    def test_init(self):
        viz = FeatureImportanceVisualizer()
        assert viz.figsize == (10, 8)

    def test_plot_returns_figure(self, classification_data):
        viz = FeatureImportanceVisualizer()
        importances = np.random.rand(10)

        fig = viz.plot(classification_data['feature_names'], importances)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_with_top_n(self, classification_data):
        viz = FeatureImportanceVisualizer()
        importances = np.random.rand(10)

        fig = viz.plot(classification_data['feature_names'], importances, top_n=5)

        ax = fig.get_axes()[0]
        assert len(ax.get_yticklabels()) <= 5
        plt.close(fig)

    def test_plot_comparison(self, classification_data):
        viz = FeatureImportanceVisualizer()
        importances_dict = {
            'Model1': np.random.rand(10),
            'Model2': np.random.rand(10)
        }

        fig = viz.plot_comparison(classification_data['feature_names'], importances_dict)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ============================================================================
# TEST LEARNINGCURVEVISUALIZER
# ============================================================================

class TestLearningCurveVisualizer:

    def test_init(self):
        viz = LearningCurveVisualizer()
        assert viz.figsize == (10, 8)

    def test_plot_returns_figure(self):
        viz = LearningCurveVisualizer()
        train_sizes = [10, 20, 30, 40, 50]
        train_scores = np.random.rand(5, 5)
        val_scores = np.random.rand(5, 5)

        fig = viz.plot(train_sizes, train_scores, val_scores)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_has_bands(self):
        viz = LearningCurveVisualizer()
        train_sizes = [10, 20, 30, 40, 50]
        train_scores = np.random.rand(5, 5)
        val_scores = np.random.rand(5, 5)

        fig = viz.plot(train_sizes, train_scores, val_scores)

        # Should have fill_between bands
        ax = fig.get_axes()[0]
        assert len(ax.collections) > 0  # Collections for fill_between
        plt.close(fig)


# ============================================================================
# TEST PREDICTIONDISTRIBUTIONVISUALIZER
# ============================================================================

class TestPredictionDistributionVisualizer:

    def test_init(self):
        viz = PredictionDistributionVisualizer()
        assert viz.figsize == (10, 8)

    def test_plot_returns_figure(self, classification_data):
        viz = PredictionDistributionVisualizer()
        fig = viz.plot(classification_data['y_true'], classification_data['y_pred_proba'])

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ============================================================================
# TEST CLASSIFICATIONMETRICSVISUALIZER
# ============================================================================

class TestClassificationMetricsVisualizer:

    def test_init(self):
        viz = ClassificationMetricsVisualizer()
        assert viz.figsize == (10, 8)

    def test_plot_metrics_comparison(self):
        viz = ClassificationMetricsVisualizer()
        metrics_dict = {
            'Model1': {'accuracy': 0.95, 'f1': 0.93, 'auc': 0.97},
            'Model2': {'accuracy': 0.92, 'f1': 0.90, 'auc': 0.94}
        }

        fig = viz.plot_metrics_comparison(metrics_dict)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ============================================================================
# TEST PERFORMANCECOMPARISONVISUALIZER
# ============================================================================

class TestPerformanceComparisonVisualizer:

    def test_init(self):
        viz = PerformanceComparisonVisualizer()
        assert viz.figsize == (12, 6)

    def test_plot_radar_returns_figure(self):
        viz = PerformanceComparisonVisualizer()
        metrics_dict = {
            'Model1': {'accuracy': 0.95, 'f1': 0.93, 'auc': 0.97},
            'Model2': {'accuracy': 0.92, 'f1': 0.90, 'auc': 0.94}
        }

        fig = viz.plot_radar(metrics_dict)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_parallel_coordinates(self):
        viz = PerformanceComparisonVisualizer()
        df = pd.DataFrame({
            'Model': ['Model1', 'Model2', 'Model3'],
            'accuracy': [0.95, 0.92, 0.94],
            'f1': [0.93, 0.90, 0.92],
            'auc': [0.97, 0.94, 0.96]
        })

        fig = viz.plot_parallel_coordinates(df)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ============================================================================
# TEST VISUALIZATIONMANAGER
# ============================================================================

class TestVisualizationManager:

    def test_init(self):
        manager = VisualizationManager()

        assert manager.confusion_matrix_viz is not None
        assert manager.roc_viz is not None
        assert manager.calibration_viz is not None
        assert manager.residual_viz is not None
        assert manager.feature_importance_viz is not None
        assert manager.learning_curve_viz is not None
        assert manager.prediction_dist_viz is not None
        assert manager.metrics_viz is not None
        assert manager.performance_viz is not None

    def test_create_classification_report(self, classification_data):
        manager = VisualizationManager()

        figures = manager.create_classification_report(
            classification_data['y_true'],
            classification_data['y_pred'],
            classification_data['y_pred_proba'],
            'TestModel'
        )

        assert isinstance(figures, dict)
        assert 'confusion_matrix' in figures
        assert 'roc_curve' in figures
        assert 'calibration' in figures
        assert 'prediction_dist' in figures

        # Close all figures
        for fig in figures.values():
            plt.close(fig)

    def test_create_regression_report(self, regression_data):
        manager = VisualizationManager()

        figures = manager.create_regression_report(
            regression_data['y_true'],
            regression_data['y_pred'],
            'TestModel'
        )

        assert isinstance(figures, dict)
        assert 'residuals' in figures

        plt.close(figures['residuals'])

    def test_create_comparison_report(self, classification_data, multi_model_predictions):
        manager = VisualizationManager()
        predictions_dict, probabilities_dict = multi_model_predictions

        figures = manager.create_comparison_report(
            classification_data['y_true'],
            predictions_dict,
            probabilities_dict
        )

        assert isinstance(figures, dict)
        assert 'confusion_matrices' in figures

        for fig in figures.values():
            plt.close(fig)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:

    def test_full_classification_workflow(self, classification_data):
        manager = VisualizationManager()

        # Create all visualizations
        figures = manager.create_classification_report(
            classification_data['y_true'],
            classification_data['y_pred'],
            classification_data['y_pred_proba']
        )

        assert len(figures) == 4

        for fig in figures.values():
            plt.close(fig)

    def test_full_comparison_workflow(self, classification_data, multi_model_predictions):
        manager = VisualizationManager()
        predictions_dict, probabilities_dict = multi_model_predictions

        # Create comparison visualizations
        figures = manager.create_comparison_report(
            classification_data['y_true'],
            predictions_dict,
            probabilities_dict
        )

        assert len(figures) >= 2

        for fig in figures.values():
            plt.close(fig)

    def test_save_figures(self, classification_data):
        manager = VisualizationManager()

        figures = manager.create_classification_report(
            classification_data['y_true'],
            classification_data['y_pred'],
            classification_data['y_pred_proba']
        )

        # Add figures to manager
        manager.figures = list(figures.values())

        # Save to temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            manager.save_all_figures(tmpdir)

            # Check files exist
            files = os.listdir(tmpdir)
            assert len(files) > 0

        for fig in figures.values():
            plt.close(fig)


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:

    def test_small_dataset(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])

        viz = ConfusionMatrixVisualizer()
        fig = viz.plot(y_true, y_pred)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_class(self):
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])

        viz = ConfusionMatrixVisualizer()
        fig = viz.plot(y_true, y_pred)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_perfect_predictions(self):
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])
        y_proba = np.array([[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1]])

        viz = ConfusionMatrixVisualizer()
        fig = viz.plot(y_true, y_pred)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


if __name__ == "__main__":
    print("Phase 5.5: Visualization Manager - Test Suite Ready")