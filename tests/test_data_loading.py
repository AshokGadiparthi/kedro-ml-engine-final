"""Tests for data loading pipeline."""

import pytest
import pandas as pd
from ml_engine.pipelines.data_loading import load_raw_data

class TestDataLoading:
    """Test data loading functionality."""
    
    def test_load_raw_data_success(self, sample_csv_file):
        """Test successful data loading."""
        df = load_raw_data(sample_csv_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert len(df.columns) == 5
    
    def test_load_raw_data_missing_file(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_raw_data("nonexistent_file.csv")
    
    def test_load_raw_data_columns(self, sample_csv_file):
        """Test column names are preserved."""
        df = load_raw_data(sample_csv_file)
        expected_cols = {'age', 'income', 'credit_score', 'employed', 'name'}
        assert set(df.columns) == expected_cols
