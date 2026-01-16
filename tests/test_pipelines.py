"""Tests for complete pipelines."""

import pytest
from kedro.pipeline import Pipeline
from ml_engine.pipelines.data_loading import create_pipeline as data_loading_pipeline
from ml_engine.pipelines.data_validation import create_pipeline as data_validation_pipeline
from ml_engine.pipelines.data_cleaning import create_pipeline as data_cleaning_pipeline

class TestPipelines:
    """Test pipeline creation and structure."""
    
    def test_data_loading_pipeline_creation(self):
        """Test data loading pipeline creation."""
        pipeline = data_loading_pipeline()
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.nodes) == 1
        assert pipeline.nodes[0].name == "load_raw_data_node"
    
    def test_data_validation_pipeline_creation(self):
        """Test data validation pipeline creation."""
        pipeline = data_validation_pipeline()
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.nodes) == 1
        assert pipeline.nodes[0].name == "validate_data_quality_node"
    
    def test_data_cleaning_pipeline_creation(self):
        """Test data cleaning pipeline creation."""
        pipeline = data_cleaning_pipeline()
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.nodes) == 1
        assert pipeline.nodes[0].name == "clean_data_node"
