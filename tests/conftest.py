"""Pytest configuration and fixtures."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    
    return pd.DataFrame({
        'age': np.random.randint(18, 80, 100),
        'income': np.random.normal(50000, 20000, 100),
        'credit_score': np.random.randint(300, 850, 100),
        'employed': np.random.choice([0, 1], 100),
        'name': [f'Person_{i}' for i in range(100)],
    })

@pytest.fixture
def sample_dataframe_with_missing() -> pd.DataFrame:
    """Create DataFrame with missing values."""
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, 100),
        'income': np.random.normal(50000, 20000, 100),
        'credit_score': np.random.randint(300, 850, 100),
        'employed': np.random.choice([0, 1], 100),
    })
    
    df.loc[0:5, 'age'] = np.nan
    df.loc[10:15, 'income'] = np.nan
    
    return df

@pytest.fixture
def temp_data_dir() -> Path:
    """Create temporary directory for data files."""
    with TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def sample_csv_file(sample_dataframe: pd.DataFrame, temp_data_dir: Path) -> str:
    """Create sample CSV file."""
    csv_path = temp_data_dir / "sample_data.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    return str(csv_path)
