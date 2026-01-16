"""Pipeline registry."""

from typing import Dict
from kedro.pipeline import Pipeline
from ml_engine.pipelines.data_loading import create_pipeline as data_loading_pipeline
from ml_engine.pipelines.data_validation import create_pipeline as data_validation_pipeline
from ml_engine.pipelines.data_cleaning import create_pipeline as data_cleaning_pipeline
from ml_engine.pipelines.end_to_end import create_pipeline as end_to_end_pipeline

def register_pipelines() -> Dict[str, Pipeline]:
    """Register all pipelines."""
    return {
        "data_loading": data_loading_pipeline(),
        "data_validation": data_validation_pipeline(),
        "data_cleaning": data_cleaning_pipeline(),
        "__default__": end_to_end_pipeline(),
    }
