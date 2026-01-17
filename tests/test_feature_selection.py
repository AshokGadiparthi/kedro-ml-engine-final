"""Tests for feature selection pipeline."""

import pytest
import pandas as pd
import numpy as np
from ml_engine.pipelines.feature_selection import (
    calculate_correlations_node,
    select_features_by_correlation_node,
    calculate_feature_importance_node,
)


class TestFeatureSelection:
    """Test feature selection nodes."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        return pd.DataFrame({
            'feat1': np.random.randn(20),
            'feat2': np.random.randn(20),
            'feat3': np.random.randn(20),
        })

    @pytest.fixture
    def params(self):
        """Test parameters."""
        return {
            'feature_selection': {
                'method': 'correlation',
                'threshold': 0.8,
                'top_n': 2,
            }
        }

    def test_calculate_correlations(self, sample_data):
        """Test correlation calculation."""
        result = calculate_correlations_node(sample_data)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == result.shape[1]

    def test_feature_importance(self, sample_data, params):
        """Test feature importance calculation."""
        result = calculate_feature_importance_node(sample_data, params)

        assert isinstance(result, dict)
        assert len(result) == 3
        assert sum(result.values()) > 0