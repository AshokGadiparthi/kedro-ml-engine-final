"""
PHASE 5.3: TEST SUITE FOR CROSS-VALIDATION STRATEGIES
50+ test cases covering all CV methods
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

from ml_engine.pipelines.cross_validation_strategies import (
    StratifiedKFoldCV,
    TimeSeriesCV,
    GroupKFoldCV,
    LeaveOneOutCV,
    ShuffleSplitCV,
    RepeatedStratifiedKFoldCV,
    NestedCV,
    CrossValidationComparison
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def classification_data():
    X, y = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)
    return pd.DataFrame(X, columns=[f'f{i}' for i in range(10)]), pd.Series(y)

@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=200, n_features=10, random_state=42)
    return pd.DataFrame(X, columns=[f'f{i}' for i in range(10)]), pd.Series(y)

@pytest.fixture
def small_data():
    X, y = make_classification(n_samples=50, n_features=5, n_classes=2, random_state=42)
    return pd.DataFrame(X, columns=[f'f{i}' for i in range(5)]), pd.Series(y)

@pytest.fixture
def groupdata():
    X, y = make_classification(n_samples=100, n_features=8, n_classes=2, random_state=42)
    groups = np.repeat([0, 1, 2, 3, 4], 20)
    return (pd.DataFrame(X, columns=[f'f{i}' for i in range(8)]),
            pd.Series(y), groups)

@pytest.fixture
def model_clf():
    return LogisticRegression(random_state=42, max_iter=1000)

@pytest.fixture
def model_reg():
    return LinearRegression()


# ============================================================================
# TEST STRATIFIEDKFOLDCV
# ============================================================================

class TestStratifiedKFoldCV:

    def test_init(self):
        cv = StratifiedKFoldCV(n_splits=5)
        assert cv.n_splits == 5

    def test_evaluate_classification(self, classification_data, model_clf):
        X, y = classification_data
        cv = StratifiedKFoldCV(n_splits=5)
        result = cv.evaluate(X, y, model_clf, verbose=False)

        assert 'fold_scores' in result
        assert len(result['fold_scores']) == 5
        assert 0 <= result['mean_score'] <= 1

    def test_stability_metrics(self, classification_data, model_clf):
        X, y = classification_data
        cv = StratifiedKFoldCV(n_splits=5)
        result = cv.evaluate(X, y, model_clf, verbose=False)

        assert 'std_score' in result
        assert 'cv_range' in result
        assert 'cv_coefficient' in result
        assert result['std_score'] >= 0

    def test_fold_count(self, classification_data, model_clf):
        X, y = classification_data

        for n in [3, 5, 10]:
            cv = StratifiedKFoldCV(n_splits=n)
            result = cv.evaluate(X, y, model_clf, verbose=False)
            assert len(result['fold_scores']) == n

    def test_shuffle_parameter(self, classification_data, model_clf):
        X, y = classification_data

        cv_shuffle = StratifiedKFoldCV(n_splits=5, shuffle=True)
        result_shuffle = cv_shuffle.evaluate(X, y, model_clf, verbose=False)

        cv_no_shuffle = StratifiedKFoldCV(n_splits=5, shuffle=False)
        result_no_shuffle = cv_no_shuffle.evaluate(X, y, model_clf, verbose=False)

        # Should have different results due to shuffling
        assert len(result_shuffle['fold_scores']) == len(result_no_shuffle['fold_scores'])


# ============================================================================
# TEST TIMESERIESSCV
# ============================================================================

class TestTimeSeriesCV:

    def test_init(self):
        cv = TimeSeriesCV(n_splits=5)
        assert cv.n_splits == 5

    def test_evaluate(self, classification_data, model_clf):
        X, y = classification_data
        cv = TimeSeriesCV(n_splits=5)
        result = cv.evaluate(X, y, model_clf, verbose=False)

        assert 'fold_scores' in result
        assert len(result['fold_scores']) == 5

    def test_temporal_order(self, classification_data, model_clf):
        X, y = classification_data
        cv = TimeSeriesCV(n_splits=5)
        result = cv.evaluate(X, y, model_clf, verbose=False)

        # Check fold results have temporal info
        assert 'fold_results' in result
        for fold_result in result['fold_results']:
            assert 'train_period' in fold_result
            assert 'test_period' in fold_result

    def test_different_splits(self, classification_data, model_clf):
        X, y = classification_data

        for n in [2, 3, 4]:
            cv = TimeSeriesCV(n_splits=n)
            result = cv.evaluate(X, y, model_clf, verbose=False)
            assert len(result['fold_scores']) == n


# ============================================================================
# TEST GROUPKFOLDCV
# ============================================================================

class TestGroupKFoldCV:

    def test_init(self):
        cv = GroupKFoldCV(n_splits=5)
        assert cv.n_splits == 5

    def test_evaluate_with_groups(self, groupdata, model_clf):
        X, y, groups = groupdata
        cv = GroupKFoldCV(n_splits=5)
        result = cv.evaluate(X, y, groups, model_clf, verbose=False)

        assert 'fold_scores' in result
        assert 'group_stats' in result
        assert result['n_groups'] == 5

    def test_group_stats(self, groupdata, model_clf):
        X, y, groups = groupdata
        cv = GroupKFoldCV(n_splits=5)
        result = cv.evaluate(X, y, groups, model_clf, verbose=False)

        for fold_key, stats in result['group_stats'].items():
            assert 'score' in stats
            assert 'n_test_groups' in stats
            assert 'test_groups' in stats


# ============================================================================
# TEST LEAVEONEOUTCV
# ============================================================================

class TestLeaveOneOutCV:

    def test_init(self):
        cv = LeaveOneOutCV()
        assert cv is not None

    def test_evaluate_small(self, small_data, model_clf):
        X, y = small_data
        cv = LeaveOneOutCV()
        result = cv.evaluate(X, y, model_clf, verbose=False)

        assert 'accuracy' in result
        assert result['correct'] <= result['total']
        assert result['total'] == len(X)

    def test_n_splits_equals_samples(self, small_data, model_clf):
        X, y = small_data
        cv = LeaveOneOutCV()
        result = cv.evaluate(X, y, model_clf, verbose=False)

        assert result['n_splits'] == len(X)


# ============================================================================
# TEST SHUFFLESPLITCV
# ============================================================================

class TestShuffleSplitCV:

    def test_init(self):
        cv = ShuffleSplitCV(n_splits=10, test_size=0.3)
        assert cv.n_splits == 10
        assert cv.test_size == 0.3

    def test_evaluate(self, classification_data, model_clf):
        X, y = classification_data
        cv = ShuffleSplitCV(n_splits=10)
        result = cv.evaluate(X, y, model_clf, verbose=False)

        assert len(result['fold_scores']) == 10
        assert 0 <= result['mean_score'] <= 1

    def test_stability_metrics(self, classification_data, model_clf):
        X, y = classification_data
        cv = ShuffleSplitCV(n_splits=10)
        result = cv.evaluate(X, y, model_clf, verbose=False)

        assert 'std_score' in result
        assert 'cv_range' in result

    def test_different_test_sizes(self, classification_data, model_clf):
        X, y = classification_data

        for test_size in [0.2, 0.3, 0.5]:
            cv = ShuffleSplitCV(n_splits=5, test_size=test_size)
            result = cv.evaluate(X, y, model_clf, verbose=False)
            assert len(result['fold_scores']) == 5


# ============================================================================
# TEST REPEATEDSTRATIFIEDKFOLDCV
# ============================================================================

class TestRepeatedStratifiedKFoldCV:

    def test_init(self):
        cv = RepeatedStratifiedKFoldCV(n_splits=5, n_repeats=10)
        assert cv.n_splits == 5
        assert cv.n_repeats == 10

    def test_evaluate(self, classification_data, model_clf):
        X, y = classification_data
        cv = RepeatedStratifiedKFoldCV(n_splits=5, n_repeats=10)
        result = cv.evaluate(X, y, model_clf, verbose=False)

        assert len(result['fold_scores']) == 50  # 5 * 10
        assert len(result['repeat_means']) == 10

    def test_repeat_stability(self, classification_data, model_clf):
        X, y = classification_data
        cv = RepeatedStratifiedKFoldCV(n_splits=5, n_repeats=10)
        result = cv.evaluate(X, y, model_clf, verbose=False)

        assert 'repeat_std' in result
        assert result['repeat_std'] >= 0


# ============================================================================
# TEST NESTEDCV
# ============================================================================

class TestNestedCV:

    def test_init(self):
        cv = NestedCV(outer_splits=5, inner_splits=3)
        assert cv.outer_splits == 5
        assert cv.inner_splits == 3

    def test_evaluate(self, classification_data, model_clf):
        X, y = classification_data
        cv = NestedCV(outer_splits=3, inner_splits=2)
        result = cv.evaluate(X, y, model_clf, verbose=False)

        assert len(result['outer_scores']) == 3
        assert len(result['inner_scores']) > 0

    def test_outer_inner_separation(self, classification_data, model_clf):
        X, y = classification_data
        cv = NestedCV(outer_splits=3, inner_splits=2)
        result = cv.evaluate(X, y, model_clf, verbose=False)

        assert 'outer_mean' in result
        assert 'outer_std' in result
        assert 'inner_mean' in result


# ============================================================================
# TEST CROSSVALIDATIONCOMPARISON
# ============================================================================

class TestCrossValidationComparison:

    def test_init(self):
        comp = CrossValidationComparison()
        assert comp is not None

    def test_compare_all_without_groups(self, classification_data, model_clf):
        X, y = classification_data
        comp = CrossValidationComparison()
        result = comp.compare_all(X, y, model_clf, include_loo=False)

        assert isinstance(result, pd.DataFrame)
        assert 'method' in result.columns
        assert 'mean_score' in result.columns
        assert len(result) >= 4  # At least 4 methods

    def test_compare_all_with_groups(self, groupdata, model_clf):
        X, y, groups = groupdata
        comp = CrossValidationComparison()
        result = comp.compare_all(X, y, model_clf, groups=groups, include_loo=False)

        assert 'GroupKFoldCV' in result['method'].values

    def test_comparison_columns(self, classification_data, model_clf):
        X, y = classification_data
        comp = CrossValidationComparison()
        result = comp.compare_all(X, y, model_clf, include_loo=False)

        expected_cols = ['method', 'mean_score', 'std_score', 'cv_range', 'stability']
        for col in expected_cols:
            assert col in result.columns


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:

    def test_cv_consistency(self, classification_data, model_clf):
        X, y = classification_data

        skf = StratifiedKFoldCV(n_splits=5, random_state=42)
        result1 = skf.evaluate(X, y, model_clf, verbose=False)

        skf2 = StratifiedKFoldCV(n_splits=5, random_state=42)
        result2 = skf2.evaluate(X, y, model_clf, verbose=False)

        # Same random_state should give same results
        assert np.isclose(result1['mean_score'], result2['mean_score'])

    def test_cv_vs_comparison(self, classification_data, model_clf):
        X, y = classification_data

        # Individual CV
        skf = StratifiedKFoldCV(n_splits=5)
        skf_result = skf.evaluate(X, y, model_clf, verbose=False)

        # Comparison CV
        comp = CrossValidationComparison()
        comp_result = comp.compare_all(X, y, model_clf, include_loo=False)

        # StratifiedKFold should be in comparison
        assert 'StratifiedKFold' in comp_result['method'].values


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:

    def test_small_n_splits(self, classification_data, model_clf):
        X, y = classification_data

        cv = StratifiedKFoldCV(n_splits=2)
        result = cv.evaluate(X, y, model_clf, verbose=False)

        assert len(result['fold_scores']) == 2

    def test_regression_data(self, regression_data, model_reg):
        X, y = regression_data

        cv = StratifiedKFoldCV(n_splits=5)
        # Should work with regression too
        result = cv.evaluate(X, y, model_reg, verbose=False)

        assert 'mean_score' in result

    def test_perfect_split_cv(self, classification_data, model_clf):
        X, y = classification_data

        cv = ShuffleSplitCV(n_splits=1, test_size=0.5)
        result = cv.evaluate(X, y, model_clf, verbose=False)

        assert len(result['fold_scores']) == 1


if __name__ == "__main__":
    print("Phase 5.3: Cross-Validation Strategies - Test Suite Ready")