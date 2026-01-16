"""Data Loading Pipeline - Kedro 0.19.5 Compatible."""

import pandas as pd
import logging
from pathlib import Path
from kedro.pipeline import Pipeline, node
from typing import Any

logger = logging.getLogger(__name__)

def load_raw_data(data_path: str) -> pd.DataFrame:
    """Load raw data from CSV file.
    
    Args:
        data_path: Path to CSV file
        
    Returns:
        DataFrame with raw data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: If reading fails
    """
    logger.info(f"📊 Loading raw data from: {data_path}")
    
    data_path_obj = Path(data_path)
    
    if not data_path_obj.exists():
        logger.error(f"   ❌ File not found: {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    try:
        df = pd.read_csv(data_path_obj)
        logger.info(f"   ✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        logger.info(f"   Columns: {list(df.columns)}")
        return df
    except Exception as e:
        logger.error(f"   ❌ Error loading data: {str(e)}")
        raise

def create_pipeline() -> Pipeline:
    """Create data loading pipeline."""
    return Pipeline(
        [
            node(
                func=load_raw_data,
                inputs="params:data_path",
                outputs="raw_data",
                name="load_raw_data_node",
                tags=["data_loading"],
            ),
        ],
        tags=["data_loading"],
    )
