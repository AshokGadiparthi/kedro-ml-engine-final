"""Data Cleaning Pipeline."""

import pandas as pd
import numpy as np
import logging
from typing import Tuple
from kedro.pipeline import Pipeline, node

logger = logging.getLogger(__name__)

def clean_data(
    df: pd.DataFrame,
    handle_missing: str = "median",
    remove_duplicates: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """Clean data with standard preprocessing."""
    logger.info("🧹 Cleaning data...")
    
    df_clean = df.copy()
    report = {"original_shape": df.shape, "actions": []}
    
    if remove_duplicates:
        n_before = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        n_removed = n_before - len(df_clean)
        
        if n_removed > 0:
            logger.info(f"   Removed {n_removed} duplicate rows")
            report["actions"].append(f"Removed {n_removed} duplicates")
    
    logger.info(f"   Handling missing values with '{handle_missing}' strategy")
    
    if handle_missing == "drop":
        n_before = len(df_clean)
        df_clean = df_clean.dropna()
        n_dropped = n_before - len(df_clean)
        logger.info(f"   Dropped {n_dropped} rows with missing values")
    
    elif handle_missing == "median":
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
    
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().any():
            mode_val = df_clean[col].mode()
            if len(mode_val) > 0:
                df_clean[col].fillna(mode_val[0], inplace=True)
    
    report["final_shape"] = df_clean.shape
    logger.info(f"   ✅ Cleaned data shape: {df_clean.shape}")
    
    return df_clean, report

def create_pipeline() -> Pipeline:
    """Create data cleaning pipeline."""
    return Pipeline(
        [
            node(
                func=clean_data,
                inputs=["raw_data", "params:data_processing.handle_missing"],
                outputs=["cleaned_data", "data_cleaning_report"],
                name="clean_data_node",
                tags=["data_cleaning"],
            ),
        ],
        tags=["data_cleaning"],
    )
