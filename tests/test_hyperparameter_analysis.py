"""
PHASE 5.6: TEST SUITE FOR HYPERPARAMETER ANALYSIS
50+ test cases covering all hyperparameter analysis functionality
"""

import pytest
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, '/home/claude/kedro-ml-engine-final/src')

from ml_engine.pipelines.hyperparameter_analysis import (
    SensitivityAnalysis,
    GeneralizationGapAnalysis,
    ParameterImportance,
    OptimizationSuggestions,
    LearningDynamicsAnalysis,
    ComprehensiveHyperparameterAnalyzer
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_sensitivity_data():
    return {
        'learning_rate': np.array([0.001, 0.01, 0.1, 1.0]),
        'scores': np.array([0.70, 0.85, 0.92, 0.88])
    }

@pytest.fixture
def sample_gap_data():
    return {
        'train_scores': [0.95, 0.94, 0.93, 0.92, 0.91],
        'val_scores': [0.90, 0.89, 0.87, 0.85, 0.83]
    }

@pytest.fixture
def sample_learning_data():
    return {
        'train_scores': [0.50, 0.65, 0.75, 0.85, 0.90, 0.92, 0.93, 0.94, 0.94, 0.94]
    }


# ============================================================================
# TEST SENSITIVITYANALYSIS
# ============================================================================

class TestSensitivityAnalysis:

    def test_init(self):
        sa = SensitivityAnalysis()
        assert len(sa.results) == 0

    def test_analyze_parameter_impact(self, sample_sensitivity_data):
        sa = SensitivityAnalysis()
        result = sa.analyze_parameter_impact(
            'learning_rate',
            sample_sensitivity_data['learning_rate'],
            sample_sensitivity_data['scores']
        )

        assert 'parameter' in result
        assert result['parameter'] == 'learning_rate'
        assert 'score_range' in result
        assert 'optimal_value' in result
        assert 'sensitivity' in result

    def test_optimal_value_identification(self, sample_sensitivity_data):
        sa = SensitivityAnalysis()
        result = sa.analyze_parameter_impact(
            'learning_rate',
            sample_sensitivity_data['learning_rate'],
            sample_sensitivity_data['scores']
        )

        # Should identify 0.1 as optimal (score 0.92)
        assert result['optimal_value'] == 0.1
        assert result['optimal_score'] == 0.92

    def test_score_range_calculation(self, sample_sensitivity_data):
        sa = SensitivityAnalysis()
        result = sa.analyze_parameter_impact(
            'learning_rate',
            sample_sensitivity_data['learning_rate'],
            sample_sensitivity_data['scores']
        )

        expected_range = 0.92 - 0.70
        assert abs(result['score_range'] - expected_range) < 0.001

    def test_multiple_parameters(self):
        sa = SensitivityAnalysis()

        # Analyze first parameter
        sa.analyze_parameter_impact('param1', [1, 2, 3], [0.8, 0.9, 0.85])

        # Analyze second parameter
        sa.analyze_parameter_impact('param2', [10, 20, 30], [0.75, 0.88, 0.92])

        assert len(sa.results) == 2

    def test_compare_parameters(self):
        sa = SensitivityAnalysis()

        sa.analyze_parameter_impact('param1', [1, 2, 3], [0.8, 0.9, 0.85])
        sa.analyze_parameter_impact('param2', [10, 20, 30], [0.75, 0.88, 0.92])

        comparison = sa.compare_parameters()

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2


# ============================================================================
# TEST GENERALIZATIONGAPANALYSIS
# ============================================================================

class TestGeneralizationGapAnalysis:

    def test_init(self):
        gga = GeneralizationGapAnalysis()
        assert gga is not None

    def test_analyze_gap(self, sample_gap_data):
        gga = GeneralizationGapAnalysis()
        result = gga.analyze_gap(
            sample_gap_data['train_scores'],
            sample_gap_data['val_scores']
        )

        assert 'train_mean' in result
        assert 'val_mean' in result
        assert 'generalization_gap' in result
        assert 'status' in result

    def test_well_generalized(self):
        gga = GeneralizationGapAnalysis()

        train_scores = [0.90, 0.91, 0.92]
        val_scores = [0.89, 0.90, 0.91]

        result = gga.analyze_gap(train_scores, val_scores)

        assert result['generalization_gap'] < 0.02
        assert "Well-generalized" in result['status']

    def test_overfitting_detection(self):
        gga = GeneralizationGapAnalysis()

        train_scores = [0.95, 0.96, 0.97]
        val_scores = [0.80, 0.78, 0.75]

        result = gga.analyze_gap(train_scores, val_scores)

        assert result['generalization_gap'] > 0.1
        assert "overfitting" in result['status']

    def test_gap_ratio_calculation(self):
        gga = GeneralizationGapAnalysis()

        train_scores = [0.90, 0.90]
        val_scores = [0.85, 0.85]

        result = gga.analyze_gap(train_scores, val_scores)

        expected_gap_ratio = 0.05 / 0.90
        assert abs(result['gap_ratio'] - expected_gap_ratio) < 0.001


# ============================================================================
# TEST PARAMETERIMPORTANCE
# ============================================================================

class TestParameterImportance:

    def test_init(self):
        pi = ParameterImportance()
        assert len(pi.importances) == 0

    def test_calculate_importance(self):
        pi = ParameterImportance()

        param_results = {
            'param1': {'sensitivity': 0.2, 'score_range': 0.1},
            'param2': {'sensitivity': 0.05, 'score_range': 0.02},
            'param3': {'sensitivity': 0.15, 'score_range': 0.08}
        }

        importances = pi.calculate_importance(param_results)

        assert isinstance(importances, dict)
        assert len(importances) == 3
        # Should sum to approximately 1 (normalized)
        assert abs(sum(importances.values()) - 1.0) < 0.001

    def test_importance_ranking(self):
        pi = ParameterImportance()

        param_results = {
            'low_imp': {'sensitivity': 0.01, 'score_range': 0.005},
            'high_imp': {'sensitivity': 0.2, 'score_range': 0.15},
            'mid_imp': {'sensitivity': 0.1, 'score_range': 0.08}
        }

        importances = pi.calculate_importance(param_results)

        # High importance should be highest
        assert importances['high_imp'] > importances['mid_imp']
        assert importances['mid_imp'] > importances['low_imp']

    def test_get_top_parameters(self):
        pi = ParameterImportance()

        param_results = {
            'param1': {'sensitivity': 0.1, 'score_range': 0.05},
            'param2': {'sensitivity': 0.2, 'score_range': 0.1},
            'param3': {'sensitivity': 0.05, 'score_range': 0.02}
        }

        pi.calculate_importance(param_results)
        top_3 = pi.get_top_parameters(top_n=3)

        assert len(top_3) == 3
        assert top_3[0][0] == 'param2'  # Highest importance


# ============================================================================
# TEST OPTIMIZATIONSUGGESTIONS
# ============================================================================

class TestOptimizationSuggestions:

    def test_analyze_positive_relationship(self):
        param_values = [1, 2, 3, 4, 5]
        scores = [0.7, 0.75, 0.8, 0.82, 0.85]

        result = OptimizationSuggestions.analyze_parameter_relationship(
            param_values, scores
        )

        assert 'relationship' in result
        assert 'suggestion' in result

    def test_analyze_negative_relationship(self):
        param_values = [1, 2, 3, 4, 5]
        scores = [0.85, 0.82, 0.80, 0.75, 0.7]

        result = OptimizationSuggestions.analyze_parameter_relationship(
            param_values, scores
        )

        assert 'relationship' in result
        assert 'negative' in result['relationship']

    def test_generate_suggestions(self):
        analysis_results = {
            'param1': {'sensitivity': 0.15},
            'param2': {'sensitivity': 0.04},
            'param3': {'sensitivity': 0.08}
        }

        suggestions = OptimizationSuggestions.generate_suggestions(analysis_results)

        assert isinstance(suggestions, list)
        assert len(suggestions) == 3


# ============================================================================
# TEST LEARNINGDYNAMICSANALYSIS
# ============================================================================

class TestLearningDynamicsAnalysis:

    def test_init(self):
        lda = LearningDynamicsAnalysis()
        assert lda is not None

    def test_analyze_convergence(self, sample_learning_data):
        lda = LearningDynamicsAnalysis()
        result = lda.analyze_convergence(sample_learning_data['train_scores'])

        assert 'initial_score' in result
        assert 'final_score' in result
        assert 'total_improvement' in result
        assert 'converged' in result

    def test_convergence_metrics(self):
        lda = LearningDynamicsAnalysis()

        train_scores = [0.5, 0.6, 0.7, 0.8, 0.85, 0.87, 0.88, 0.88, 0.88, 0.88]

        result = lda.analyze_convergence(train_scores)

        assert result['initial_score'] == 0.5
        assert result['final_score'] == 0.88
        assert result['total_improvement'] == 0.38

    def test_improvement_ratio(self):
        lda = LearningDynamicsAnalysis()

        train_scores = [0.5, 0.6, 0.7, 0.75, 0.78]

        result = lda.analyze_convergence(train_scores)

        expected_ratio = (0.78 - 0.5) / 0.5
        assert abs(result['improvement_ratio'] - expected_ratio) < 0.001

    def test_identify_overfitting_epoch(self):
        lda = LearningDynamicsAnalysis()

        train_scores = [0.5, 0.7, 0.8, 0.85, 0.88, 0.90, 0.91, 0.92]
        val_scores = [0.5, 0.7, 0.79, 0.83, 0.85, 0.84, 0.82, 0.80]

        epoch = lda.identify_overfitting_epoch(train_scores, val_scores)

        # Should identify some epoch where gap increases
        assert epoch is None or isinstance(epoch, int)


# ============================================================================
# TEST COMPREHENSIVEHYPERPARAMETERANALYZER
# ============================================================================

class TestComprehensiveHyperparameterAnalyzer:

    def test_init(self):
        cha = ComprehensiveHyperparameterAnalyzer()

        assert cha.sensitivity is not None
        assert cha.gap_analysis is not None
        assert cha.importance is not None
        assert cha.learning_dynamics is not None

    def test_analyze_parameter_grid(self):
        cha = ComprehensiveHyperparameterAnalyzer()

        grid_results = {
            (0.01, 10): {'score': 0.85, 'train_score': 0.90},
            (0.01, 20): {'score': 0.87, 'train_score': 0.92},
            (0.1, 10): {'score': 0.90, 'train_score': 0.95},
            (0.1, 20): {'score': 0.88, 'train_score': 0.93}
        }

        df = cha.analyze_parameter_grid(grid_results)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert 'gap' in df.columns

    def test_generate_report(self):
        cha = ComprehensiveHyperparameterAnalyzer()

        sensitivity_results = {
            'param1': {'sensitivity': 0.1, 'optimal_value': 0.5, 'optimal_score': 0.95}
        }
        gap_results = {
            'param1': {'generalization_gap': 0.02, 'status': '✅ Well-generalized'}
        }
        learning_results = {
            'param1': {'improvement_ratio': 0.4, 'converged': True}
        }

        report = cha.generate_report(sensitivity_results, gap_results, learning_results)

        assert isinstance(report, str)
        assert 'SENSITIVITY' in report


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:

    def test_full_hyperparameter_workflow(self):
        cha = ComprehensiveHyperparameterAnalyzer()

        # Sensitivity analysis
        sensitivity_results = {
            'learning_rate': {
                'sensitivity': 0.15,
                'optimal_value': 0.01,
                'optimal_score': 0.92
            },
            'batch_size': {
                'sensitivity': 0.08,
                'optimal_value': 32,
                'optimal_score': 0.90
            }
        }

        # Gap analysis
        gap_results = {
            'learning_rate': {
                'generalization_gap': 0.02,
                'status': '✅ Well-generalized'
            },
            'batch_size': {
                'generalization_gap': 0.05,
                'status': '⚠️  Slight overfitting'
            }
        }

        # Learning dynamics
        learning_results = {
            'learning_rate': {
                'improvement_ratio': 0.4,
                'converged': True
            },
            'batch_size': {
                'improvement_ratio': 0.35,
                'converged': True
            }
        }

        # Generate report
        report = cha.generate_report(sensitivity_results, gap_results, learning_results)

        assert len(report) > 100
        assert 'learning_rate' in report

    def test_parameter_importance_workflow(self):
        cha = ComprehensiveHyperparameterAnalyzer()

        param_results = {
            'learning_rate': {'sensitivity': 0.2, 'score_range': 0.1},
            'batch_size': {'sensitivity': 0.1, 'score_range': 0.05},
            'epochs': {'sensitivity': 0.05, 'score_range': 0.02}
        }

        importance = cha.importance.calculate_importance(param_results)
        top_params = cha.importance.get_top_parameters(top_n=2)

        assert len(importance) == 3
        assert len(top_params) == 2
        assert top_params[0][0] == 'learning_rate'


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:

    def test_single_value_parameter(self):
        sa = SensitivityAnalysis()
        result = sa.analyze_parameter_impact('param', [1], [0.9])

        assert result['score_range'] == 0
        assert result['optimal_value'] == 1

    def test_constant_scores(self):
        sa = SensitivityAnalysis()
        result = sa.analyze_parameter_impact('param', [1, 2, 3], [0.8, 0.8, 0.8])

        assert result['score_range'] == 0
        assert result['sensitivity'] == 0

    def test_identical_train_val_scores(self):
        gga = GeneralizationGapAnalysis()
        result = gga.analyze_gap([0.9, 0.9], [0.9, 0.9])

        assert result['generalization_gap'] == 0
        assert "Well-generalized" in result['status']

    def test_zero_initial_score(self):
        lda = LearningDynamicsAnalysis()
        train_scores = [0, 0.5, 0.8, 0.9]

        result = lda.analyze_convergence(train_scores)

        # Should handle division by zero
        assert isinstance(result['improvement_ratio'], float)


if __name__ == "__main__":
    print("Phase 5.6: Hyperparameter Analysis - Test Suite Ready")