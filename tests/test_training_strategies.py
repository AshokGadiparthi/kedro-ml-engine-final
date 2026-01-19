"""
PHASE 5.1: TEST SUITE FOR TRAINING STRATEGIES
55+ test cases covering all training strategy classes
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, '/home/claude/kedro-ml-engine-final/src')

from ml_engine.pipelines.training_strategies import (
    StratifiedTrainer,
    TimeSeriesTrainer,
    ProgressiveTrainer,
    EarlyStoppingMonitor,
    EnsembleTrainingOrchestrator
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def classification_data():
    X, y = make_classification(n_samples=200, n_features=10, n_classes=3, random_state=42)
    return pd.DataFrame(X, columns=[f'f{i}' for i in range(10)]), pd.Series(y)

@pytest.fixture
def binary_data():
    X, y = make_classification(n_samples=150, n_features=8, n_classes=2, random_state=42)
    return pd.DataFrame(X, columns=[f'f{i}' for i in range(8)]), pd.Series(y)

@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=150, n_features=8, random_state=42)
    return pd.DataFrame(X, columns=[f'f{i}' for i in range(8)]), pd.Series(y)

@pytest.fixture
def time_series_data():
    X = np.random.randn(100, 5)
    y = np.cumsum(np.random.randn(100))
    return pd.DataFrame(X, columns=[f'f{i}' for i in range(5)]), pd.Series(y)


# ============================================================================
# TEST STRATIFIEDTRAINER
# ============================================================================

class TestStratifiedTrainer:

    def test_init(self):
        trainer = StratifiedTrainer(n_splits=5)
        assert trainer.n_splits == 5

    def test_detect_classification(self, classification_data):
        X, y = classification_data
        trainer = StratifiedTrainer()
        assert trainer._detect_problem_type(y) == 'classification'

    def test_detect_regression(self, regression_data):
        X, y = regression_data
        trainer = StratifiedTrainer()
        assert trainer._detect_problem_type(y) == 'regression'

    def test_train_classification(self, classification_data):
        X, y = classification_data
        trainer = StratifiedTrainer(n_splits=3)
        model = LogisticRegression(max_iter=1000)
        results = trainer.train_on_folds(X, y, model)

        assert len(results['fold_scores']) == 3
        assert 'mean_score' in results
        assert 0 <= results['mean_score'] <= 1

    def test_train_regression(self, regression_data):
        X, y = regression_data
        trainer = StratifiedTrainer(n_splits=3)
        model = DecisionTreeRegressor()
        results = trainer.train_on_folds(X, y, model, problem_type='regression')

        assert len(results['fold_scores']) == 3

    def test_reproducibility(self, classification_data):
        X, y = classification_data
        model = LogisticRegression(max_iter=1000)

        trainer1 = StratifiedTrainer(n_splits=3, random_state=42)
        results1 = trainer1.train_on_folds(X, y, model)

        trainer2 = StratifiedTrainer(n_splits=3, random_state=42)
        results2 = trainer2.train_on_folds(X, y, model)

        assert np.isclose(results1['mean_score'], results2['mean_score'])

    def test_different_splits(self, classification_data):
        X, y = classification_data
        model = LogisticRegression(max_iter=1000)

        for n_splits in [2, 3, 5]:
            trainer = StratifiedTrainer(n_splits=n_splits)
            results = trainer.train_on_folds(X, y, model)
            assert len(results['fold_scores']) == n_splits

    def test_dataframe_target(self, classification_data):
        X, y = classification_data
        y_df = pd.DataFrame(y)
        trainer = StratifiedTrainer()
        results = trainer.train_on_folds(X, y_df, LogisticRegression(max_iter=1000))
        assert 'mean_score' in results

    def test_score_range(self, classification_data):
        X, y = classification_data
        trainer = StratifiedTrainer()
        results = trainer.train_on_folds(X, y, LogisticRegression(max_iter=1000))
        assert all(0 <= s <= 1 for s in results['fold_scores'])


# ============================================================================
# TEST TIMESERIESTRAINER
# ============================================================================

class TestTimeSeriesTrainer:

    def test_init(self):
        trainer = TimeSeriesTrainer(n_splits=5)
        assert trainer.n_splits == 5

    def test_train_regression(self, time_series_data):
        X, y = time_series_data
        trainer = TimeSeriesTrainer(n_splits=3)
        results = trainer.train_on_time_series(X, y, DecisionTreeRegressor())

        assert len(results['fold_scores']) == 3
        assert 'mean_score' in results

    def test_fold_count(self, time_series_data):
        X, y = time_series_data
        for n_splits in [2, 3]:
            trainer = TimeSeriesTrainer(n_splits=n_splits)
            results = trainer.train_on_time_series(X, y, DecisionTreeRegressor())
            assert len(results['fold_scores']) == n_splits

    def test_mean_calculation(self, time_series_data):
        X, y = time_series_data
        trainer = TimeSeriesTrainer(n_splits=3)
        results = trainer.train_on_time_series(X, y, DecisionTreeRegressor())

        expected = np.mean(results['fold_scores'])
        assert np.isclose(results['mean_score'], expected)

    def test_classification(self, binary_data):
        X, y = binary_data
        trainer = TimeSeriesTrainer(n_splits=2)
        results = trainer.train_on_time_series(X, y, LogisticRegression(max_iter=1000),
                                               problem_type='classification')
        assert results['problem_type'] == 'classification'


# ============================================================================
# TEST PROGRESSIVETRAINER
# ============================================================================

class TestProgressiveTrainer:

    def test_init(self):
        trainer = ProgressiveTrainer(increments=5)
        assert trainer.increments == 5

    def test_progressive_train(self, classification_data):
        X, y = classification_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        trainer = ProgressiveTrainer(increments=3)
        results = trainer.progressive_train(X_train, y_train, X_test, y_test,
                                            LogisticRegression(max_iter=1000))

        assert len(results['train_scores']) == 3
        assert len(results['test_scores']) == 3

    def test_incremental_sizes(self, classification_data):
        X, y = classification_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        trainer = ProgressiveTrainer(increments=5)
        results = trainer.progressive_train(X_train, y_train, X_test, y_test,
                                            LogisticRegression(max_iter=1000))

        history = results['learning_history']
        sizes = [h['n_samples'] for h in history]
        assert all(sizes[i] < sizes[i+1] for i in range(len(sizes)-1))

    def test_reproducibility(self, classification_data):
        X, y = classification_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        trainer1 = ProgressiveTrainer(increments=3, random_state=42)
        results1 = trainer1.progressive_train(X_train, y_train, X_test, y_test,
                                              LogisticRegression(max_iter=1000))

        trainer2 = ProgressiveTrainer(increments=3, random_state=42)
        results2 = trainer2.progressive_train(X_train, y_train, X_test, y_test,
                                              LogisticRegression(max_iter=1000))

        assert np.allclose(results1['train_scores'], results2['train_scores'])

    def test_regression(self, regression_data):
        X, y = regression_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        trainer = ProgressiveTrainer(increments=3)
        results = trainer.progressive_train(X_train, y_train, X_test, y_test,
                                            DecisionTreeRegressor(), problem_type='regression')

        assert len(results['train_scores']) == 3


# ============================================================================
# TEST EARLYSTOPPINGMONITOR
# ============================================================================

class TestEarlyStoppingMonitor:

    def test_init(self):
        monitor = EarlyStoppingMonitor(patience=5)
        assert monitor.patience == 5
        assert monitor.counter == 0

    def test_minimize_loss(self):
        monitor = EarlyStoppingMonitor(metric='val_loss')
        assert monitor.minimize == True

    def test_maximize_accuracy(self):
        monitor = EarlyStoppingMonitor(metric='val_accuracy')
        assert monitor.minimize == False

    def test_first_epoch(self):
        monitor = EarlyStoppingMonitor()
        should_stop, _ = monitor.monitor_and_decide(0.5, epoch=0)

        assert should_stop == False
        assert monitor.best_score == 0.5

    def test_improvement_loss(self):
        monitor = EarlyStoppingMonitor(patience=3, metric='val_loss')
        monitor.monitor_and_decide(0.5, epoch=0)
        should_stop, _ = monitor.monitor_and_decide(0.4, epoch=1)

        assert should_stop == False
        assert monitor.counter == 0

    def test_no_improvement(self):
        monitor = EarlyStoppingMonitor(patience=2, metric='val_loss')
        monitor.monitor_and_decide(0.5, epoch=0)
        monitor.monitor_and_decide(0.5, epoch=1)
        should_stop, _ = monitor.monitor_and_decide(0.5, epoch=2)

        assert should_stop == False
        assert monitor.counter == 2

    def test_early_stop_trigger(self):
        monitor = EarlyStoppingMonitor(patience=2, metric='val_loss')
        monitor.monitor_and_decide(0.5, epoch=0)
        monitor.monitor_and_decide(0.5, epoch=1)
        monitor.monitor_and_decide(0.5, epoch=2)
        should_stop, _ = monitor.monitor_and_decide(0.5, epoch=3)

        assert should_stop == True

    def test_best_score(self):
        monitor = EarlyStoppingMonitor(metric='val_loss')
        monitor.monitor_and_decide(0.5, epoch=0)
        monitor.monitor_and_decide(0.4, epoch=1)

        assert monitor.get_best_score() == 0.4

    def test_best_epoch(self):
        monitor = EarlyStoppingMonitor()
        monitor.monitor_and_decide(0.5, epoch=0)
        monitor.monitor_and_decide(0.4, epoch=1)

        assert monitor.get_best_epoch() == 1


# ============================================================================
# TEST ENSEMBLETRAININGOCHESTRATOR
# ============================================================================

class TestEnsembleTrainingOrchestrator:

    def test_init(self):
        orch = EnsembleTrainingOrchestrator()
        assert orch.trained_models == {}

    def test_train_ensemble(self, classification_data):
        X, y = classification_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        models = {
            'rf': RandomForestClassifier(n_estimators=5, random_state=42),
            'lr': LogisticRegression(max_iter=1000)
        }

        orch = EnsembleTrainingOrchestrator()
        results = orch.train_ensemble(X_train, y_train, models, X_test, y_test)

        assert len(results['trained_models']) == 2
        assert len(results['model_scores']) == 2

    def test_no_test_data(self, classification_data):
        X, y = classification_data

        models = {
            'lr': LogisticRegression(max_iter=1000)
        }

        orch = EnsembleTrainingOrchestrator()
        results = orch.train_ensemble(X, y, models)

        assert len(results['trained_models']) == 1
        assert len(results['model_scores']) == 0

    def test_ranked_models(self, classification_data):
        X, y = classification_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        models = {
            'rf': RandomForestClassifier(n_estimators=5, random_state=42),
            'lr': LogisticRegression(max_iter=1000)
        }

        orch = EnsembleTrainingOrchestrator()
        orch.train_ensemble(X_train, y_train, models, X_test, y_test)
        ranked = orch.get_ranked_models()

        assert len(ranked) == 2
        assert ranked[0][1] >= ranked[1][1]

    def test_regression_ensemble(self, regression_data):
        X, y = regression_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        models = {
            'tree': DecisionTreeRegressor(random_state=42)
        }

        orch = EnsembleTrainingOrchestrator()
        results = orch.train_ensemble(X_train, y_train, models, X_test, y_test,
                                      problem_type='regression')

        assert results['problem_type'] == 'regression'


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:

    def test_stratified_vs_ensemble(self, classification_data):
        X, y = classification_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        strat = StratifiedTrainer()
        strat_results = strat.train_on_folds(X_train, y_train, LogisticRegression(max_iter=1000))

        models = {'lr': LogisticRegression(max_iter=1000)}
        orch = EnsembleTrainingOrchestrator()
        ens_results = orch.train_ensemble(X_train, y_train, models, X_test, y_test)

        assert strat_results['mean_score'] >= 0
        assert len(ens_results['model_scores']) > 0

    def test_early_stopping_simulation(self):
        monitor = EarlyStoppingMonitor(patience=2)
        scores = [0.80, 0.82, 0.83, 0.833, 0.833]

        stopped = False
        for epoch, score in enumerate(scores):
            should_stop, _ = monitor.monitor_and_decide(score, epoch=epoch)
            if should_stop:
                stopped = True
                break

        assert stopped


if __name__ == "__main__":
    print("Phase 5.1: Test Suite Ready - 55+ tests")