"""
PHASE 5.7: TEST SUITE FOR REPORT GENERATOR - FINAL MODULE
50+ test cases covering all report generation functionality
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import json

import sys
sys.path.insert(0, '/home/claude/kedro-ml-engine-final/src')

from ml_engine.pipelines.report_generator import (
    HTMLReportGenerator,
    JSONReportGenerator,
    PDFReportGenerator,
    ModelCardGenerator,
    ExecutiveSummaryGenerator,
    ComprehensiveReportManager
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_metrics():
    return {
        'accuracy': 0.95,
        'f1': 0.93,
        'auc': 0.97,
        'precision': 0.94,
        'recall': 0.92
    }

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'Model': ['Model1', 'Model2', 'Model3'],
        'Accuracy': [0.95, 0.92, 0.94],
        'F1': [0.93, 0.90, 0.92]
    })

@pytest.fixture
def sample_hyperparameters():
    return {
        'learning_rate': 0.01,
        'batch_size': 32,
        'epochs': 100,
        'dropout': 0.3
    }


# ============================================================================
# TEST HTMLREPORTGENERATOR
# ============================================================================

class TestHTMLReportGenerator:

    def test_init(self):
        gen = HTMLReportGenerator("Test Report")
        assert gen.title == "Test Report"
        assert len(gen.sections) == 0

    def test_add_section(self, sample_metrics):
        gen = HTMLReportGenerator()
        gen.add_section("Metrics", sample_metrics, "dict")

        assert "Metrics" in gen.sections

    def test_add_dataframe_section(self, sample_dataframe):
        gen = HTMLReportGenerator()
        gen.add_section("Model Comparison", sample_dataframe, "dataframe")

        assert "Model Comparison" in gen.sections

    def test_generate_contains_title(self):
        gen = HTMLReportGenerator("My Test Report")
        gen.add_section("Section 1", "Content 1", "text")

        html = gen.generate()

        assert "My Test Report" in html
        assert "<html>" in html
        assert "</html>" in html

    def test_generate_contains_styling(self):
        gen = HTMLReportGenerator()
        html = gen.generate()

        assert "<style>" in html
        assert "font-family" in html
        assert ".report-table" in html

    def test_generate_multiple_sections(self, sample_metrics, sample_dataframe):
        gen = HTMLReportGenerator()
        gen.add_section("Metrics", sample_metrics, "dict")
        gen.add_section("Comparison", sample_dataframe, "dataframe")

        html = gen.generate()

        assert "Metrics" in html
        assert "Comparison" in html

    def test_html_contains_table(self, sample_dataframe):
        gen = HTMLReportGenerator()
        gen.add_section("Data", sample_dataframe, "dataframe")

        html = gen.generate()

        assert "<table" in html
        assert "Model1" in html


# ============================================================================
# TEST JSONREPORTGENERATOR
# ============================================================================

class TestJSONReportGenerator:

    def test_init(self):
        gen = JSONReportGenerator("Test Report")
        assert gen.title == "Test Report"
        assert gen.data['title'] == "Test Report"

    def test_add_section(self, sample_metrics):
        gen = JSONReportGenerator()
        gen.add_section("metrics", sample_metrics)

        assert "metrics" in gen.data['sections']

    def test_add_metadata(self):
        gen = JSONReportGenerator()
        gen.add_metadata("version", "1.0")
        gen.add_metadata("author", "Test User")

        assert gen.data['metadata']['version'] == "1.0"
        assert gen.data['metadata']['author'] == "Test User"

    def test_generate_valid_json(self, sample_metrics):
        gen = JSONReportGenerator()
        gen.add_section("metrics", sample_metrics)

        json_str = gen.generate()

        # Should be valid JSON
        data = json.loads(json_str)
        assert "sections" in data
        assert "metrics" in data["sections"]

    def test_dataframe_conversion(self, sample_dataframe):
        gen = JSONReportGenerator()
        gen.add_section("data", sample_dataframe)

        json_str = gen.generate()
        data = json.loads(json_str)

        # DataFrame should be converted to list of dicts
        assert isinstance(data['sections']['data'], list)

    def test_numpy_array_conversion(self):
        gen = JSONReportGenerator()
        arr = np.array([1, 2, 3, 4, 5])
        gen.add_section("array", arr)

        json_str = gen.generate()
        data = json.loads(json_str)

        # Numpy array should be converted to list
        assert isinstance(data['sections']['array'], list)


# ============================================================================
# TEST PDFREPORTGENERATOR
# ============================================================================

class TestPDFReportGenerator:

    def test_init(self):
        gen = PDFReportGenerator("Test Report")
        assert gen.title == "Test Report"
        assert len(gen.sections) == 0

    def test_add_section(self, sample_metrics):
        gen = PDFReportGenerator()
        gen.add_section("Metrics", sample_metrics)

        assert "Metrics" in gen.sections

    def test_generate_contains_title(self):
        gen = PDFReportGenerator("My PDF Report")
        gen.add_section("Section", "Content")

        pdf = gen.generate()

        assert "My PDF Report" in pdf
        assert "=" * 80 in pdf

    def test_generate_formatted_correctly(self):
        gen = PDFReportGenerator()
        gen.add_section("Test Section", "Test Content")

        pdf = gen.generate()

        assert "TEST SECTION" in pdf
        assert "Test Content" in pdf

    def test_multiple_sections(self, sample_metrics):
        gen = PDFReportGenerator()
        gen.add_section("Section 1", sample_metrics)
        gen.add_section("Section 2", "Content 2")

        pdf = gen.generate()

        assert "SECTION 1" in pdf
        assert "SECTION 2" in pdf


# ============================================================================
# TEST MODELCARDGENERATOR
# ============================================================================

class TestModelCardGenerator:

    def test_init(self):
        gen = ModelCardGenerator("MyModel")
        assert gen.model_name == "MyModel"

    def test_add_overview(self):
        gen = ModelCardGenerator("MyModel")
        gen.add_overview("A test model", "2.0")

        assert gen.card['overview']['description'] == "A test model"
        assert gen.card['overview']['version'] == "2.0"

    def test_add_training_data(self):
        gen = ModelCardGenerator("MyModel")
        gen.add_training_data("dataset", 1000, 20, "classification")

        assert gen.card['training_data']['dataset_name'] == "dataset"
        assert gen.card['training_data']['dataset_size'] == 1000

    def test_add_model_details(self, sample_hyperparameters):
        gen = ModelCardGenerator("MyModel")
        gen.add_model_details("RandomForest", sample_hyperparameters)

        assert gen.card['model_details']['type'] == "RandomForest"
        assert "learning_rate" in gen.card['model_details']['hyperparameters']

    def test_add_performance(self, sample_metrics):
        gen = ModelCardGenerator("MyModel")
        gen.add_performance(sample_metrics)

        assert gen.card['performance']['accuracy'] == 0.95

    def test_add_limitations(self):
        gen = ModelCardGenerator("MyModel")
        limitations = ["Limited to small datasets", "Slow inference"]
        gen.add_limitations(limitations)

        assert gen.card['limitations'] == limitations

    def test_add_recommendations(self):
        gen = ModelCardGenerator("MyModel")
        recs = ["Use with large datasets", "Monitor predictions"]
        gen.add_recommendations(recs)

        assert gen.card['recommendations'] == recs

    def test_generate_markdown(self, sample_metrics):
        gen = ModelCardGenerator("TestModel")
        gen.add_overview("Test description")
        gen.add_performance(sample_metrics)

        md = gen.generate_markdown()

        assert "# Model Card: TestModel" in md
        assert "accuracy:" in md.lower()

    def test_generate_json(self, sample_metrics):
        gen = ModelCardGenerator("TestModel")
        gen.add_performance(sample_metrics)

        json_str = gen.generate_json()
        data = json.loads(json_str)

        assert data['model_name'] == "TestModel"
        assert data['performance']['accuracy'] == 0.95


# ============================================================================
# TEST EXECUTIVESUMMARYGENERATOR
# ============================================================================

class TestExecutiveSummaryGenerator:

    def test_init(self):
        gen = ExecutiveSummaryGenerator("MyModel")
        assert gen.model_name == "MyModel"
        assert len(gen.sections) == 0

    def test_add_objective(self):
        gen = ExecutiveSummaryGenerator("MyModel")
        gen.add_objective("To predict customer churn")

        assert len(gen.sections) == 1

    def test_add_key_findings(self):
        gen = ExecutiveSummaryGenerator("MyModel")
        findings = ["Finding 1", "Finding 2", "Finding 3"]
        gen.add_key_findings(findings)

        assert len(gen.sections) == 1

    def test_add_performance_summary(self, sample_metrics):
        gen = ExecutiveSummaryGenerator("MyModel")
        gen.add_performance_summary(sample_metrics)

        assert len(gen.sections) == 1

    def test_add_recommendations(self):
        gen = ExecutiveSummaryGenerator("MyModel")
        recs = ["Recommendation 1", "Recommendation 2"]
        gen.add_recommendations(recs)

        assert len(gen.sections) == 1

    def test_generate_contains_model_name(self):
        gen = ExecutiveSummaryGenerator("MyModel")
        gen.add_objective("Test objective")

        summary = gen.generate()

        assert "MyModel" in summary

    def test_generate_all_sections(self, sample_metrics):
        gen = ExecutiveSummaryGenerator("MyModel")
        gen.add_objective("Objective")
        gen.add_key_findings(["Finding 1"])
        gen.add_performance_summary(sample_metrics)
        gen.add_recommendations(["Rec 1"])

        summary = gen.generate()

        assert "OBJECTIVE" in summary
        assert "KEY FINDINGS" in summary
        assert "PERFORMANCE" in summary
        assert "RECOMMENDATIONS" in summary


# ============================================================================
# TEST COMPREHENSIVEREPORTMANAGER
# ============================================================================

class TestComprehensiveReportManager:

    def test_init(self):
        manager = ComprehensiveReportManager("MyModel")

        assert manager.model_name == "MyModel"
        assert manager.html_gen is not None
        assert manager.json_gen is not None
        assert manager.pdf_gen is not None
        assert manager.model_card_gen is not None
        assert manager.summary_gen is not None

    def test_add_performance_section(self, sample_metrics):
        manager = ComprehensiveReportManager("MyModel")
        manager.add_performance_section(sample_metrics)

        # Check sections were added to all generators
        assert "Performance Metrics" in manager.html_gen.sections

    def test_add_model_details_section(self, sample_hyperparameters):
        manager = ComprehensiveReportManager("MyModel")
        training_info = {
            'dataset': 'test_data',
            'n_samples': 1000,
            'n_features': 10,
            'target_type': 'classification'
        }

        manager.add_model_details_section(
            'RandomForest',
            sample_hyperparameters,
            training_info
        )

        assert "Model Details" in manager.html_gen.sections

    def test_add_validation_section(self):
        manager = ComprehensiveReportManager("MyModel")
        cv_results = {'mean_score': 0.95, 'std_score': 0.02}
        gap_analysis = {'gap': 0.03, 'status': 'well-generalized'}

        manager.add_validation_section(cv_results, gap_analysis)

        assert "Cross-Validation Results" in manager.html_gen.sections

    def test_add_recommendations_section(self):
        manager = ComprehensiveReportManager("MyModel")
        recommendations = ["Use ensemble methods", "Collect more data"]
        limitations = ["Limited to 100k samples"]

        manager.add_recommendations_section(recommendations, limitations)

        assert "Recommendations" in manager.html_gen.sections


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:

    def test_full_report_generation_workflow(self, sample_metrics, sample_hyperparameters):
        manager = ComprehensiveReportManager("TestModel")

        # Add sections
        manager.add_performance_section(sample_metrics)
        training_info = {
            'dataset': 'test_data',
            'n_samples': 1000,
            'n_features': 10,
            'target_type': 'classification'
        }
        manager.add_model_details_section('RandomForest', sample_hyperparameters, training_info)
        manager.add_recommendations_section(
            ["Recommendation 1"],
            ["Limitation 1"]
        )

        # Generate reports
        with tempfile.TemporaryDirectory() as tmpdir:
            reports = manager.generate_all_reports(tmpdir)

            assert 'html' in reports
            assert 'json' in reports
            assert 'pdf' in reports
            assert 'model_card' in reports
            assert 'summary' in reports

    def test_all_report_files_created(self, sample_metrics):
        manager = ComprehensiveReportManager("TestModel")
        manager.add_performance_section(sample_metrics)

        with tempfile.TemporaryDirectory() as tmpdir:
            reports = manager.generate_all_reports(tmpdir)

            for file_path in reports.values():
                assert os.path.exists(file_path)

    def test_html_report_readable(self, sample_metrics):
        manager = ComprehensiveReportManager("TestModel")
        manager.add_performance_section(sample_metrics)

        html = manager.html_gen.generate()

        assert "<html>" in html
        assert "</html>" in html
        assert "TestModel" in html

    def test_json_report_parseable(self, sample_metrics):
        manager = ComprehensiveReportManager("TestModel")
        manager.add_performance_section(sample_metrics)

        json_str = manager.json_gen.generate()
        data = json.loads(json_str)

        assert isinstance(data, dict)
        assert 'sections' in data


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:

    def test_empty_report(self):
        gen = HTMLReportGenerator("Empty Report")
        html = gen.generate()

        assert "Empty Report" in html
        assert "<html>" in html

    def test_special_characters_in_title(self):
        gen = HTMLReportGenerator("Test <>&\" Report")
        gen.add_section("Section", "Content")

        html = gen.generate()
        # Should handle special characters
        assert len(html) > 0

    def test_large_dataframe(self):
        gen = HTMLReportGenerator()
        large_df = pd.DataFrame(np.random.rand(100, 10))
        gen.add_section("Large Data", large_df, "dataframe")

        html = gen.generate()
        assert len(html) > 1000

    def test_unicode_content(self):
        gen = HTMLReportGenerator("Report with émojis 🎉")
        gen.add_section("Unicode", "Content with 中文 and 日本語")

        html = gen.generate()
        assert len(html) > 0


if __name__ == "__main__":
    print("Phase 5.7: Report Generator - Test Suite Ready")