"""Data Cleaning Pipeline - Kedro 0.19.5 Compatible."""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any
from kedro.pipeline import Pipeline, node

logger = logging.getLogger(__name__)

def clean_data(
    df: pd.DataFrame,
    handle_missing: str = "median",
    remove_duplicates: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean data with standard preprocessing.
    
    Args:
        df: DataFrame to clean
        handle_missing: Strategy for missing values (drop, median, mean)
        remove_duplicates: Whether to remove duplicate rows
        
    Returns:
        Tuple of (cleaned DataFrame, report dictionary)
    """
    logger.info("🧹 Cleaning data...")
    
    df_clean = df.copy()
    report: Dict[str, Any] = {"original_shape": df.shape, "actions": []}
    
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
        report["actions"].append(f"Dropped {n_dropped} rows with missing values")
    
    elif handle_missing == "median":
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
        
        if len(numeric_cols) > 0:
            logger.info(f"   Filled {len(numeric_cols)} numeric columns with median")
            report["actions"].append(f"Filled {len(numeric_cols)} numeric columns with median")
    
    elif handle_missing == "mean":
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                mean_val = df_clean[col].mean()
                df_clean[col].fillna(mean_val, inplace=True)
    
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().any():
            mode_val = df_clean[col].mode()
            if len(mode_val) > 0:
                df_clean[col].fillna(mode_val[0], inplace=True)
    
    report["final_shape"] = df_clean.shape
    report["rows_removed"] = df.shape[0] - df_clean.shape[0]
    
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
