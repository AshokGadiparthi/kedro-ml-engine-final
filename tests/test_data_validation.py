"""Tests for data validation."""

import pytest
from ml_engine.utils.validators import validate_dataframe, validate_X_y
from ml_engine.utils.exceptions import DataValidationError, InsufficientDataError
import pandas as pd
import numpy as np

class TestDataValidation:
    """Test data validation functionality."""
    
    def test_validate_dataframe_success(self, sample_dataframe: pd.DataFrame) -> None:
        """Test successful validation."""
        report = validate_dataframe(sample_dataframe)
        
        assert report["valid"] is True
        assert len(report["errors"]) == 0
    
    def test_validate_dataframe_too_few_rows(self) -> None:
        """Test validation with too few rows."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        report = validate_dataframe(df, min_rows=10)
        
        assert report["valid"] is False
        assert len(report["errors"]) > 0
    
    def test_validate_dataframe_empty_column(self) -> None:
        """Test validation with empty column."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [np.nan, np.nan, np.nan]})
        report = validate_dataframe(df)
        
        assert report["valid"] is False
    
    def test_validate_X_y_success(self) -> None:
        """Test successful X and y validation."""
        X = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        y = pd.Series([0, 1, 0, 1, 0])
        
        assert validate_X_y(X, y) is True
    
    def test_validate_X_y_length_mismatch(self) -> None:
        """Test validation with mismatched lengths."""
        X = pd.DataFrame({'a': [1, 2, 3]})
        y = pd.Series([0, 1])
        
        with pytest.raises(DataValidationError):
            validate_X_y(X, y)
    
    def test_validate_X_y_insufficient_data(self) -> None:
        """Test validation with insufficient data."""
        X = pd.DataFrame({'a': [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        
        with pytest.raises(InsufficientDataError):
            validate_X_y(X, y)
