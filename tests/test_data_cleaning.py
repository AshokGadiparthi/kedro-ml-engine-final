"""Tests for data cleaning pipeline."""

import pytest
import pandas as pd
import numpy as np
from ml_engine.pipelines.data_cleaning import clean_data

class TestDataCleaning:
    """Test data cleaning functionality."""
    
    def test_clean_data_remove_duplicates(self) -> None:
        """Test duplicate removal."""
        df = pd.DataFrame({
            'a': [1, 2, 2, 3],
            'b': [4, 5, 5, 6],
        })
        
        cleaned, report = clean_data(df, remove_duplicates=True, handle_missing='drop')
        
        assert len(cleaned) == 3
        assert report["actions"][0] == "Removed 1 duplicates"
    
    def test_clean_data_handle_missing_median(self) -> None:
        """Test missing value handling with median."""
        df = pd.DataFrame({
            'a': [1.0, 2.0, np.nan, 4.0, 5.0],
            'b': [10.0, 20.0, 30.0, np.nan, 50.0],
        })
        
        cleaned, report = clean_data(df, handle_missing='median', remove_duplicates=False)
        
        assert cleaned['a'].isna().sum() == 0
        assert cleaned['b'].isna().sum() == 0
    
    def test_clean_data_handle_missing_drop(self) -> None:
        """Test missing value handling with drop."""
        df = pd.DataFrame({
            'a': [1, 2, np.nan, 4],
            'b': [5, 6, 7, 8],
        })
        
        cleaned, report = clean_data(df, handle_missing='drop', remove_duplicates=False)
        
        assert len(cleaned) == 3
        assert cleaned['a'].isna().sum() == 0
