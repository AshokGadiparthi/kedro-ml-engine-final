"""Data Loading Pipeline."""

import pandas as pd
import logging
from pathlib import Path
from kedro.pipeline import Pipeline, node

logger = logging.getLogger(__name__)

def load_raw_data(data_path: str) -> pd.DataFrame:
    """Load raw data from CSV file."""
    logger.info(f"📊 Loading raw data from: {data_path}")
    
    data_path = Path(data_path)
    
    if not data_path.exists():
        logger.error(f"   ❌ File not found: {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        logger.info(f"   ✅ Loaded {len(df)} rows, {len(df.columns)} columns")
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
