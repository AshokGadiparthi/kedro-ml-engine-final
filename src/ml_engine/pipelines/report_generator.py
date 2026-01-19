"""
PHASE 5.7: COMPREHENSIVE REPORT GENERATION
===========================================
Automated report generation for ML models:
- HTML report generation with styling
- JSON export for integration
- PDF reports (text-based)
- Model cards (standardized format)
- Executive summaries
- Complete documentation

Status: PRODUCTION READY
Lines: 700+ core implementation
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

log = logging.getLogger(__name__)


# ============================================================================
# HTML REPORT GENERATOR
# ============================================================================

class HTMLReportGenerator:
    """Generate HTML reports with styling."""

    def __init__(self, title: str = "ML Model Report"):
        """Initialize HTML report generator."""
        self.title = title
        self.sections = {}

        log.info(f"✅ HTMLReportGenerator initialized: {title}")

    def add_section(self, section_name: str, content: Any, section_type: str = "text"):
        """Add section to report."""
        self.sections[section_name] = {
            'content': content,
            'type': section_type
        }

    def _generate_table_html(self, df: pd.DataFrame) -> str:
        """Generate HTML table from DataFrame."""
        return df.to_html(classes='report-table', index=True)

    def _generate_html_header(self) -> str:
        """Generate HTML header with styling."""
        header = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 40px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            margin-bottom: 15px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }}
        h3 {{
            color: #7f8c8d;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        .metadata {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 30px;
            font-size: 0.9em;
        }}
        .metric {{
            display: inline-block;
            margin-right: 30px;
            margin-bottom: 10px;
        }}
        .metric-label {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        .metric-value {{
            color: #2c3e50;
            font-size: 1.2em;
            font-weight: bold;
        }}
        .report-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .report-table th {{
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        .report-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid #ecf0f1;
        }}
        .report-table tr:hover {{
            background-color: #f8f9fa;
        }}
        .success {{
            color: #27ae60;
            font-weight: bold;
        }}
        .warning {{
            color: #f39c12;
            font-weight: bold;
        }}
        .danger {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
            text-align: center;
            color: #95a5a6;
            font-size: 0.9em;
        }}
        .code-block {{
            background-color: #f4f4f4;
            border-left: 3px solid #3498db;
            padding: 10px;
            margin: 10px 0;
            font-family: 'Courier New', monospace;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{self.title}</h1>
        <div class="metadata">
            <div class="metric">
                <div class="metric-label">Generated:</div>
                <div class="metric-value">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
            </div>
        </div>
"""
        return header

    def _generate_html_footer(self) -> str:
        """Generate HTML footer."""
        footer = """
        <div class="footer">
            <p>Generated by ML Engine Report Generator | Phase 5 Complete</p>
        </div>
    </div>
</body>
</html>
"""
        return footer

    def generate(self) -> str:
        """Generate complete HTML report."""
        html = self._generate_html_header()

        for section_name, section_data in self.sections.items():
            content = section_data['content']
            section_type = section_data['type']

            html += f"\n        <h2>{section_name}</h2>\n"

            if section_type == 'dataframe' and isinstance(content, pd.DataFrame):
                html += self._generate_table_html(content)
            elif section_type == 'dict':
                html += "        <div class='code-block'>\n"
                for key, value in content.items():
                    html += f"            <strong>{key}:</strong> {value}<br>\n"
                html += "        </div>\n"
            else:
                html += f"        <p>{str(content)}</p>\n"

        html += self._generate_html_footer()

        log.info(f"✅ HTML report generated: {len(html)} characters")

        return html


# ============================================================================
# JSON REPORT GENERATOR
# ============================================================================

class JSONReportGenerator:
    """Generate JSON reports for data integration."""

    def __init__(self, title: str = "ML Model Report"):
        """Initialize JSON report generator."""
        self.title = title
        self.data = {
            'title': title,
            'timestamp': datetime.now().isoformat(),
            'sections': {}
        }

        log.info(f"✅ JSONReportGenerator initialized: {title}")

    def add_section(self, section_name: str, content: Any):
        """Add section to JSON report."""
        # Convert DataFrame to dict
        if isinstance(content, pd.DataFrame):
            content = content.to_dict('records')
        # Convert numpy arrays
        elif isinstance(content, np.ndarray):
            content = content.tolist()

        self.data['sections'][section_name] = content

    def add_metadata(self, key: str, value: Any):
        """Add metadata to report."""
        if 'metadata' not in self.data:
            self.data['metadata'] = {}

        self.data['metadata'][key] = value

    def generate(self) -> str:
        """Generate JSON report."""
        log.info(f"✅ JSON report generated")

        return json.dumps(self.data, indent=2, default=str)

    def save(self, filename: str):
        """Save JSON report to file."""
        with open(filename, 'w') as f:
            f.write(self.generate())

        log.info(f"✅ JSON report saved: {filename}")


# ============================================================================
# PDF REPORT GENERATOR
# ============================================================================

class PDFReportGenerator:
    """Generate PDF reports (text-based)."""

    def __init__(self, title: str = "ML Model Report"):
        """Initialize PDF report generator."""
        self.title = title
        self.sections = {}

        log.info(f"✅ PDFReportGenerator initialized: {title}")

    def add_section(self, section_name: str, content: Any):
        """Add section to report."""
        self.sections[section_name] = content

    def _format_section(self, section_name: str, content: Any) -> str:
        """Format section for PDF."""
        lines = []
        lines.append("\n" + "=" * 80)
        lines.append(section_name.upper())
        lines.append("=" * 80 + "\n")

        if isinstance(content, pd.DataFrame):
            lines.append(content.to_string())
        elif isinstance(content, dict):
            for key, value in content.items():
                lines.append(f"{key}: {value}")
        else:
            lines.append(str(content))

        return "\n".join(lines)

    def generate(self) -> str:
        """Generate PDF-style text report."""
        lines = []

        # Header
        lines.append("=" * 80)
        lines.append(self.title.center(80))
        lines.append("=" * 80)
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Sections
        for section_name, content in self.sections.items():
            lines.append(self._format_section(section_name, content))

        # Footer
        lines.append("\n" + "=" * 80)
        lines.append("END OF REPORT".center(80))
        lines.append("=" * 80)

        report = "\n".join(lines)

        log.info(f"✅ PDF report generated: {len(report)} characters")

        return report


# ============================================================================
# MODEL CARD GENERATOR
# ============================================================================

class ModelCardGenerator:
    """Generate standardized model cards."""

    def __init__(self, model_name: str):
        """Initialize model card generator."""
        self.model_name = model_name
        self.card = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat()
        }

        log.info(f"✅ ModelCardGenerator initialized: {model_name}")

    def add_overview(self, description: str, version: str = "1.0"):
        """Add model overview."""
        self.card['overview'] = {
            'description': description,
            'version': version
        }

    def add_training_data(self, dataset_name: str, size: int,
                          features: int, target_type: str):
        """Add training data information."""
        self.card['training_data'] = {
            'dataset_name': dataset_name,
            'dataset_size': size,
            'features': features,
            'target_type': target_type
        }

    def add_model_details(self, model_type: str, hyperparameters: Dict[str, Any]):
        """Add model details."""
        self.card['model_details'] = {
            'type': model_type,
            'hyperparameters': hyperparameters
        }

    def add_performance(self, metrics: Dict[str, float]):
        """Add performance metrics."""
        self.card['performance'] = metrics

    def add_limitations(self, limitations: List[str]):
        """Add model limitations."""
        self.card['limitations'] = limitations

    def add_recommendations(self, recommendations: List[str]):
        """Add usage recommendations."""
        self.card['recommendations'] = recommendations

    def generate_markdown(self) -> str:
        """Generate model card in Markdown format."""
        md = f"# Model Card: {self.model_name}\n\n"
        md += f"**Generated:** {self.card['timestamp']}\n\n"

        if 'overview' in self.card:
            md += "## Overview\n"
            md += f"{self.card['overview']['description']}\n"
            md += f"**Version:** {self.card['overview']['version']}\n\n"

        if 'training_data' in self.card:
            md += "## Training Data\n"
            td = self.card['training_data']
            md += f"- **Dataset:** {td['dataset_name']}\n"
            md += f"- **Size:** {td['dataset_size']} samples\n"
            md += f"- **Features:** {td['features']}\n"
            md += f"- **Target Type:** {td['target_type']}\n\n"

        if 'model_details' in self.card:
            md += "## Model Details\n"
            md += f"- **Type:** {self.card['model_details']['type']}\n"
            md += "- **Hyperparameters:**\n"
            for param, value in self.card['model_details']['hyperparameters'].items():
                md += f"  - {param}: {value}\n"
            md += "\n"

        if 'performance' in self.card:
            md += "## Performance\n"
            for metric, value in self.card['performance'].items():
                md += f"- **{metric}:** {value:.4f}\n"
            md += "\n"

        if 'limitations' in self.card:
            md += "## Limitations\n"
            for limitation in self.card['limitations']:
                md += f"- {limitation}\n"
            md += "\n"

        if 'recommendations' in self.card:
            md += "## Recommendations\n"
            for rec in self.card['recommendations']:
                md += f"- {rec}\n"

        log.info(f"✅ Model card generated for {self.model_name}")

        return md

    def generate_json(self) -> str:
        """Generate model card in JSON format."""
        return json.dumps(self.card, indent=2, default=str)


# ============================================================================
# EXECUTIVE SUMMARY GENERATOR
# ============================================================================

class ExecutiveSummaryGenerator:
    """Generate executive summaries of model performance."""

    def __init__(self, model_name: str):
        """Initialize executive summary generator."""
        self.model_name = model_name
        self.sections = []

        log.info(f"✅ ExecutiveSummaryGenerator initialized: {model_name}")

    def add_objective(self, objective: str):
        """Add objective section."""
        self.sections.append(('OBJECTIVE', objective))

    def add_key_findings(self, findings: List[str]):
        """Add key findings."""
        findings_text = "\n".join(f"• {f}" for f in findings)
        self.sections.append(('KEY FINDINGS', findings_text))

    def add_performance_summary(self, metrics: Dict[str, float]):
        """Add performance summary."""
        metrics_text = "\n".join(f"• {k}: {v:.4f}" for k, v in metrics.items())
        self.sections.append(('PERFORMANCE', metrics_text))

    def add_recommendations(self, recommendations: List[str]):
        """Add recommendations."""
        rec_text = "\n".join(f"{i+1}. {r}" for i, r in enumerate(recommendations))
        self.sections.append(('RECOMMENDATIONS', rec_text))

    def generate(self) -> str:
        """Generate executive summary."""
        log.info("=" * 80)
        log.info(f"📊 EXECUTIVE SUMMARY: {self.model_name}")
        log.info("=" * 80)

        summary = f"EXECUTIVE SUMMARY: {self.model_name}\n"
        summary += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        for section_name, content in self.sections:
            summary += f"{section_name}:\n"
            summary += "-" * 40 + "\n"
            summary += content + "\n\n"

        log.info(summary)
        log.info("=" * 80)

        return summary


# ============================================================================
# COMPREHENSIVE REPORT MANAGER
# ============================================================================

class ComprehensiveReportManager:
    """Master class for comprehensive report generation."""

    def __init__(self, model_name: str):
        """Initialize comprehensive report manager."""
        self.model_name = model_name
        self.html_gen = HTMLReportGenerator(f"ML Model Report: {model_name}")
        self.json_gen = JSONReportGenerator(f"ML Model Report: {model_name}")
        self.pdf_gen = PDFReportGenerator(f"ML Model Report: {model_name}")
        self.model_card_gen = ModelCardGenerator(model_name)
        self.summary_gen = ExecutiveSummaryGenerator(model_name)

        log.info(f"✅ ComprehensiveReportManager initialized: {model_name}")

    def add_performance_section(self, metrics: Dict[str, float],
                                comparison_df: Optional[pd.DataFrame] = None):
        """Add performance section to all reports."""

        # HTML
        self.html_gen.add_section("Performance Metrics", metrics, "dict")
        if comparison_df is not None:
            self.html_gen.add_section("Model Comparison", comparison_df, "dataframe")

        # JSON
        self.json_gen.add_section("performance_metrics", metrics)
        if comparison_df is not None:
            self.json_gen.add_section("model_comparison", comparison_df)

        # PDF
        self.pdf_gen.add_section("Performance Metrics",
                                 pd.Series(metrics).to_frame())

        # Summary
        self.summary_gen.add_performance_summary(metrics)

    def add_model_details_section(self, model_type: str,
                                  hyperparameters: Dict[str, Any],
                                  training_info: Dict[str, Any]):
        """Add model details section."""

        details = {
            'type': model_type,
            'hyperparameters': hyperparameters,
            'training_info': training_info
        }

        # HTML
        self.html_gen.add_section("Model Details", details, "dict")

        # JSON
        self.json_gen.add_section("model_details", details)

        # PDF
        self.pdf_gen.add_section("Model Details",
                                 pd.Series(details).to_frame())

        # Model Card
        self.model_card_gen.add_model_details(model_type, hyperparameters)
        self.model_card_gen.add_training_data(
            training_info.get('dataset', 'Unknown'),
            training_info.get('n_samples', 0),
            training_info.get('n_features', 0),
            training_info.get('target_type', 'Unknown')
        )

    def add_validation_section(self, cv_results: Dict[str, Any],
                               gap_analysis: Dict[str, Any]):
        """Add validation section."""

        validation_data = {
            'cv_results': cv_results,
            'gap_analysis': gap_analysis
        }

        # HTML
        self.html_gen.add_section("Cross-Validation Results", cv_results, "dict")
        self.html_gen.add_section("Generalization Gap", gap_analysis, "dict")

        # JSON
        self.json_gen.add_section("validation", validation_data)

        # PDF
        self.pdf_gen.add_section("Cross-Validation Results",
                                 pd.Series(cv_results).to_frame())

    def add_recommendations_section(self, recommendations: List[str],
                                    limitations: List[str]):
        """Add recommendations section."""

        # HTML
        self.html_gen.add_section("Recommendations", recommendations, "text")
        self.html_gen.add_section("Limitations", limitations, "text")

        # JSON
        self.json_gen.add_section("recommendations", recommendations)
        self.json_gen.add_section("limitations", limitations)

        # PDF
        self.pdf_gen.add_section("Recommendations",
                                 pd.DataFrame({'Recommendation': recommendations}))

        # Model Card
        self.model_card_gen.add_recommendations(recommendations)
        self.model_card_gen.add_limitations(limitations)

        # Summary
        self.summary_gen.add_recommendations(recommendations)

    def generate_all_reports(self, output_dir: str = "./reports") -> Dict[str, str]:
        """Generate all report types."""
        import os

        os.makedirs(output_dir, exist_ok=True)

        reports = {}

        log.info("\n" + "=" * 80)
        log.info("📊 GENERATING ALL REPORTS")
        log.info("=" * 80)

        # HTML Report
        html_content = self.html_gen.generate()
        html_file = f"{output_dir}/{self.model_name}_report.html"
        with open(html_file, 'w') as f:
            f.write(html_content)
        reports['html'] = html_file
        log.info(f"✅ HTML Report: {html_file}")

        # JSON Report
        json_content = self.json_gen.generate()
        json_file = f"{output_dir}/{self.model_name}_report.json"
        with open(json_file, 'w') as f:
            f.write(json_content)
        reports['json'] = json_file
        log.info(f"✅ JSON Report: {json_file}")

        # PDF Report
        pdf_content = self.pdf_gen.generate()
        pdf_file = f"{output_dir}/{self.model_name}_report.txt"
        with open(pdf_file, 'w') as f:
            f.write(pdf_content)
        reports['pdf'] = pdf_file
        log.info(f"✅ PDF Report: {pdf_file}")

        # Model Card
        card_content = self.model_card_gen.generate_markdown()
        card_file = f"{output_dir}/{self.model_name}_modelcard.md"
        with open(card_file, 'w') as f:
            f.write(card_content)
        reports['model_card'] = card_file
        log.info(f"✅ Model Card: {card_file}")

        # Executive Summary
        summary_content = self.summary_gen.generate()
        summary_file = f"{output_dir}/{self.model_name}_executive_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(summary_content)
        reports['summary'] = summary_file
        log.info(f"✅ Executive Summary: {summary_file}")

        log.info("=" * 80)

        return reports


if __name__ == "__main__":
    print("✅ Phase 5.7: Report Generator - READY")