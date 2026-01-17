"""Tests for feature engineering pipeline."""

import pytest
import pandas as pd
import numpy as np
from ml_engine.pipelines.feature_engineering import (
    handle_missing_values_node,
    scale_features_node,
    create_interaction_features_node,
    generate_feature_statistics_node,
)


class TestFeatureEngineering:
    """Test feature engineering nodes."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with missing values."""
        return pd.DataFrame({
            'col1': [1.0, 2.0, np.nan, 4.0, 5.0],
            'col2': [10.0, 20.0, 30.0, np.nan, 50.0],
            'col3': [100.0, 200.0, 300.0, 400.0, 500.0],
        })

    @pytest.fixture
    def params(self):
        """Test parameters."""
        return {
            'missing_value_strategy': {'method': 'mean'},
            'scaling': {'method': 'standard', 'with_mean': True, 'with_std': True},
            'feature_engineering': {'create_interactions': False, 'create_polynomial': False},
        }

    def test_handle_missing_values_mean(self, sample_data, params):
        """Test mean imputation."""
        result = handle_missing_values_node(sample_data, params)

        assert result.isnull().sum().sum() == 0
        assert result.shape == sample_data.shape

    def test_scale_features_standard(self, sample_data, params):
        """Test standard scaling."""
        imputed = handle_missing_values_node(sample_data, params)
        result = scale_features_node(imputed, params)

        assert np.allclose(result.mean().mean(), 0, atol=1e-10)
        assert np.allclose(result.std().mean(), 1, atol=0.1)

    def test_create_interaction_features(self, sample_data, params):
        """Test interaction feature creation."""
        imputed = handle_missing_values_node(sample_data, params)
        scaled = scale_features_node(imputed, params)

        params['feature_engineering']['create_interactions'] = True
        result = create_interaction_features_node(scaled, params)

        assert result.shape[1] > scaled.shape[1]