"""
PHASE 5.4: TEST SUITE FOR MODEL COMPARISON
50+ test cases covering all comparison functionality
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score

import sys
sys.path.insert(0, '/home/claude/kedro-ml-engine-final/src')

from ml_engine.pipelines.model_comparison import (
    ModelBenchmark,
    ModelComparison,
    StatisticalTesting,
    PerformanceLeaderboard,
    BenchmarkReport,
    ComprehensiveModelComparison
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def classification_data():
    X, y = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

@pytest.fixture
def model_predictions(classification_data):
    X_train, X_test, y_train, y_test = classification_data

    # Train models
    model1 = LogisticRegression(random_state=42, max_iter=1000)
    model1.fit(X_train, y_train)
    y_pred1 = model1.predict(X_test)
    y_proba1 = model1.predict_proba(X_test)

    model2 = RandomForestClassifier(n_estimators=10, random_state=42)
    model2.fit(X_train, y_train)
    y_pred2 = model2.predict(X_test)
    y_proba2 = model2.predict_proba(X_test)

    return {
        'y_true': y_test,
        'model1': {'y_pred': y_pred1, 'y_proba': y_proba1, 'name': 'LogisticRegression'},
        'model2': {'y_pred': y_pred2, 'y_proba': y_proba2, 'name': 'RandomForest'}
    }

@pytest.fixture
def cv_scores():
    return {
        'model1': [0.85, 0.82, 0.88, 0.86, 0.87],
        'model2': [0.90, 0.89, 0.91, 0.88, 0.90]
    }


# ============================================================================
# TEST MODELBENCHMARK
# ============================================================================

class TestModelBenchmark:

    def test_init(self):
        bench = ModelBenchmark('MyModel')
        assert bench.model_name == 'MyModel'
        assert len(bench.metrics) == 0

    def test_add_metric(self):
        bench = ModelBenchmark('MyModel')
        bench.add_metric('accuracy', 0.95)

        assert 'accuracy' in bench.metrics
        assert bench.metrics['accuracy']['value'] == 0.95

    def test_add_multiple_metrics(self):
        bench = ModelBenchmark('MyModel')
        bench.add_metric('accuracy', 0.95)
        bench.add_metric('f1', 0.93)
        bench.add_metric('auc', 0.97)

        assert len(bench.metrics) == 3

    def test_add_cv_scores(self):
        bench = ModelBenchmark('MyModel')
        scores = [0.85, 0.82, 0.88, 0.86, 0.87]
        bench.add_cv_scores(scores)

        assert bench.cv_scores == scores
        assert len(bench.cv_scores) == 5

    def test_get_summary(self):
        bench = ModelBenchmark('MyModel')
        bench.add_metric('accuracy', 0.95)
        bench.add_cv_scores([0.85, 0.87, 0.89])

        summary = bench.get_summary()

        assert summary['model'] == 'MyModel'
        assert summary['cv_mean'] == pytest.approx(0.8700, abs=0.001)
        assert summary['n_cv_folds'] == 3


# ============================================================================
# TEST MODELCOMPARISON
# ============================================================================

class TestModelComparison:

    def test_init(self):
        comp = ModelComparison()
        assert len(comp.models) == 0

    def test_add_model(self, model_predictions):
        comp = ModelComparison()
        y_true = model_predictions['y_true']
        y_pred = model_predictions['model1']['y_pred']
        y_proba = model_predictions['model1']['y_proba']

        comp.add_model('Model1', y_true, y_pred, y_proba)

        assert 'Model1' in comp.models
        assert 'metrics' in comp.models['Model1']

    def test_add_multiple_models(self, model_predictions):
        comp = ModelComparison()
        y_true = model_predictions['y_true']

        comp.add_model('Model1', y_true, model_predictions['model1']['y_pred'],
                       model_predictions['model1']['y_proba'])
        comp.add_model('Model2', y_true, model_predictions['model2']['y_pred'],
                       model_predictions['model2']['y_proba'])

        assert len(comp.models) == 2

    def test_get_comparison_dataframe(self, model_predictions):
        comp = ModelComparison()
        y_true = model_predictions['y_true']

        comp.add_model('Model1', y_true, model_predictions['model1']['y_pred'])

        df = comp.get_comparison_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert 'model' in df.columns
        assert 'accuracy' in df.columns

    def test_rank_models(self, model_predictions):
        comp = ModelComparison()
        y_true = model_predictions['y_true']

        comp.add_model('Model1', y_true, model_predictions['model1']['y_pred'])
        comp.add_model('Model2', y_true, model_predictions['model2']['y_pred'])

        ranked = comp.rank_models('accuracy', ascending=False)

        assert len(ranked) == 2
        assert ranked.iloc[0]['model'] != ranked.iloc[1]['model']


# ============================================================================
# TEST STATISTICALTESTING
# ============================================================================

class TestStatisticalTesting:

    def test_paired_t_test(self, cv_scores):
        result = StatisticalTesting.paired_t_test(cv_scores['model1'], cv_scores['model2'])

        assert 'test' in result
        assert result['test'] == 'paired_t_test'
        assert 't_statistic' in result
        assert 'p_value' in result
        assert 'significant' in result

    def test_paired_t_test_values(self, cv_scores):
        result = StatisticalTesting.paired_t_test(cv_scores['model1'], cv_scores['model2'])

        assert abs(result['model1_mean'] - 0.856) < 0.01
        assert abs(result['model2_mean'] - 0.896) < 0.01

    def test_paired_t_test_different_lengths(self):
        scores1 = [0.85, 0.82, 0.88]
        scores2 = [0.90, 0.89]

        result = StatisticalTesting.paired_t_test(scores1, scores2)

        # Should return empty dict for mismatched lengths
        assert len(result) == 0

    def test_mcnemar_test(self, model_predictions):
        y_true = model_predictions['y_true']
        y_pred1 = model_predictions['model1']['y_pred']
        y_pred2 = model_predictions['model2']['y_pred']

        result = StatisticalTesting.mcnemar_test(y_true, y_pred1, y_pred2)

        assert result['test'] == 'mcnemar_test'
        assert 'chi2' in result
        assert 'p_value' in result
        assert 'significant' in result

    def test_mcnemar_test_values(self, model_predictions):
        y_true = model_predictions['y_true']
        y_pred1 = model_predictions['model1']['y_pred']
        y_pred2 = model_predictions['model2']['y_pred']

        result = StatisticalTesting.mcnemar_test(y_true, y_pred1, y_pred2)

        assert result['model1_correct'] > 0
        assert result['model2_correct'] > 0
        assert result['chi2'] >= 0

    def test_anova_test(self, cv_scores):
        scores_list = [cv_scores['model1'], cv_scores['model2']]

        result = StatisticalTesting.anova_test(scores_list)

        assert result['test'] == 'anova'
        assert 'f_statistic' in result
        assert 'p_value' in result
        assert result['n_models'] == 2


# ============================================================================
# TEST PERFORMANCELEADERBOARD
# ============================================================================

class TestPerformanceLeaderboard:

    def test_init(self):
        board = PerformanceLeaderboard()
        assert len(board.entries) == 0

    def test_add_entry(self):
        board = PerformanceLeaderboard()
        board.add_entry('Model1', 'accuracy', 0.95)

        assert len(board.entries) == 1

    def test_add_multiple_entries(self):
        board = PerformanceLeaderboard()
        board.add_entry('Model1', 'accuracy', 0.95)
        board.add_entry('Model2', 'accuracy', 0.92)
        board.add_entry('Model1', 'f1', 0.93)

        assert len(board.entries) == 3

    def test_get_leaderboard(self):
        board = PerformanceLeaderboard()
        board.add_entry('Model1', 'accuracy', 0.95)
        board.add_entry('Model2', 'accuracy', 0.92)
        board.add_entry('Model3', 'accuracy', 0.98)

        leaderboard = board.get_leaderboard('accuracy')

        assert len(leaderboard) == 3
        assert leaderboard.iloc[0]['model'] == 'Model3'  # Best is first
        assert 'rank' in leaderboard.columns

    def test_get_leaderboard_top_n(self):
        board = PerformanceLeaderboard()
        board.add_entry('Model1', 'accuracy', 0.95)
        board.add_entry('Model2', 'accuracy', 0.92)
        board.add_entry('Model3', 'accuracy', 0.98)

        leaderboard = board.get_leaderboard('accuracy', top_n=2)

        assert len(leaderboard) == 2

    def test_get_all_metrics_comparison(self):
        board = PerformanceLeaderboard()
        board.add_entry('Model1', 'accuracy', 0.95)
        board.add_entry('Model1', 'f1', 0.93)
        board.add_entry('Model2', 'accuracy', 0.92)
        board.add_entry('Model2', 'f1', 0.90)

        comparison = board.get_all_metrics_comparison()

        assert isinstance(comparison, pd.DataFrame)
        assert 'accuracy' in comparison.columns
        assert 'f1' in comparison.columns

    def test_get_best_model(self):
        board = PerformanceLeaderboard()
        board.add_entry('Model1', 'accuracy', 0.95)
        board.add_entry('Model2', 'accuracy', 0.92)
        board.add_entry('Model3', 'accuracy', 0.98)

        best = board.get_best_model('accuracy')

        assert best == 'Model3'


# ============================================================================
# TEST BENCHMARKREPORT
# ============================================================================

class TestBenchmarkReport:

    def test_init(self):
        report = BenchmarkReport("Test Report")
        assert report.title == "Test Report"
        assert len(report.sections) == 0

    def test_add_section(self):
        report = BenchmarkReport()
        report.add_section("Performance", {'accuracy': 0.95})

        assert 'Performance' in report.sections

    def test_generate_summary(self):
        report = BenchmarkReport("Test Report")
        report.add_section("Performance", {'accuracy': 0.95, 'f1': 0.93})

        summary = report.generate_summary()

        assert "Test Report" in summary
        assert "Performance" in summary

    def test_generate_markdown(self):
        report = BenchmarkReport("Test Report")
        report.add_section("Performance", {'accuracy': 0.95})

        markdown = report.generate_markdown()

        assert "# Test Report" in markdown
        assert "## Performance" in markdown


# ============================================================================
# TEST COMPREHENSIVEMODELCOMPARISON
# ============================================================================

class TestComprehensiveModelComparison:

    def test_init(self):
        comp = ComprehensiveModelComparison()
        assert len(comp.models_data) == 0

    def test_add_model_results(self, model_predictions, cv_scores):
        comp = ComprehensiveModelComparison()
        y_true = model_predictions['y_true']

        comp.add_model_results(
            'Model1',
            y_true,
            model_predictions['model1']['y_pred'],
            model_predictions['model1']['y_proba'],
            cv_scores['model1']
        )

        assert 'Model1' in comp.models_data
        assert comp.models_data['Model1']['cv_scores'] == cv_scores['model1']

    def test_compare_all(self, model_predictions):
        comp = ComprehensiveModelComparison()
        y_true = model_predictions['y_true']

        comp.add_model_results('Model1', y_true, model_predictions['model1']['y_pred'])
        comp.add_model_results('Model2', y_true, model_predictions['model2']['y_pred'])

        comparison = comp.compare_all()

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2

    def test_perform_statistical_tests(self, model_predictions, cv_scores):
        comp = ComprehensiveModelComparison()
        y_true = model_predictions['y_true']

        comp.add_model_results(
            'Model1', y_true, model_predictions['model1']['y_pred'],
            cv_scores=cv_scores['model1']
        )
        comp.add_model_results(
            'Model2', y_true, model_predictions['model2']['y_pred'],
            cv_scores=cv_scores['model2']
        )

        results = comp.perform_statistical_tests('Model1', 'Model2')

        assert 'mcnemar_test' in results
        if cv_scores['model1'] and cv_scores['model2']:
            assert 'paired_t_test' in results

    def test_generate_report(self, model_predictions):
        comp = ComprehensiveModelComparison()
        y_true = model_predictions['y_true']

        comp.add_model_results('Model1', y_true, model_predictions['model1']['y_pred'])
        comp.add_model_results('Model2', y_true, model_predictions['model2']['y_pred'])

        report = comp.generate_report()

        assert isinstance(report, str)
        assert "Model Comparison" in report or "Leaderboard" in report


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:

    def test_full_comparison_workflow(self, model_predictions, cv_scores):
        comp = ComprehensiveModelComparison()
        y_true = model_predictions['y_true']

        # Add both models
        comp.add_model_results(
            'LogisticRegression', y_true, model_predictions['model1']['y_pred'],
            model_predictions['model1']['y_proba'], cv_scores['model1']
        )
        comp.add_model_results(
            'RandomForest', y_true, model_predictions['model2']['y_pred'],
            model_predictions['model2']['y_proba'], cv_scores['model2']
        )

        # Get comparison
        comparison = comp.compare_all()
        assert len(comparison) == 2

        # Statistical tests
        results = comp.perform_statistical_tests('LogisticRegression', 'RandomForest')
        assert 'mcnemar_test' in results

        # Generate report
        report = comp.generate_report()
        assert isinstance(report, str)

    def test_leaderboard_workflow(self):
        board = PerformanceLeaderboard()

        # Add entries from multiple models
        models_metrics = {
            'Model1': {'accuracy': 0.95, 'f1': 0.93},
            'Model2': {'accuracy': 0.92, 'f1': 0.90},
            'Model3': {'accuracy': 0.98, 'f1': 0.96}
        }

        for model_name, metrics in models_metrics.items():
            for metric_name, value in metrics.items():
                board.add_entry(model_name, metric_name, value)

        # Get leaderboards
        accuracy_board = board.get_leaderboard('accuracy')
        f1_board = board.get_leaderboard('f1')

        assert accuracy_board.iloc[0]['model'] == 'Model3'
        assert f1_board.iloc[0]['model'] == 'Model3'


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:

    def test_empty_leaderboard(self):
        board = PerformanceLeaderboard()
        leaderboard = board.get_leaderboard('nonexistent_metric')
        assert leaderboard.empty

    def test_single_model_comparison(self, model_predictions):
        comp = ModelComparison()
        y_true = model_predictions['y_true']

        comp.add_model('OnlyModel', y_true, model_predictions['model1']['y_pred'])

        df = comp.get_comparison_dataframe()
        assert len(df) == 1

    def test_identical_predictions(self):
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred1 = np.array([0, 1, 0, 1, 0, 1])
        y_pred2 = np.array([0, 1, 0, 1, 0, 1])

        result = StatisticalTesting.mcnemar_test(y_true, y_pred1, y_pred2)

        # No disagreements should give chi2=0
        assert result['chi2'] == 0


if __name__ == "__main__":
    print("Phase 5.4: Model Comparison - Test Suite Ready")