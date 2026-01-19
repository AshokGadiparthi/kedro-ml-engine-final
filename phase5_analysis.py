#!/usr/bin/env python3
"""
════════════════════════════════════════════════════════════════════════════
PHASE 5: ADVANCED ANALYSIS & REPORTING
════════════════════════════════════════════════════════════════════════════

This script uses Phase 5 modules to analyze Phase 4 outputs.
Run this AFTER: kedro run --pipeline complete

Usage:
  python3 phase5_analysis.py

Or in Jupyter:
  %run phase5_analysis.py
"""

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("\n" + "="*80)
print("PHASE 5: ADVANCED ANALYSIS & REPORTING")
print("="*80 + "\n")

# ════════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD PHASE 4 OUTPUTS
# ════════════════════════════════════════════════════════════════════════════

print("📂 Step 1: Loading Phase 4 outputs...\n")

data_dir = Path("data/07_model_output")
output_dir = Path("data/08_reporting")
output_dir.mkdir(parents=True, exist_ok=True)

try:
    # Load Phase 4 results
    with open(data_dir / "phase4_results_with_ensemble.pkl", "rb") as f:
        phase4_results = pickle.load(f)
    print(f"✅ Loaded Phase 4 results")

    # Try to load models
    try:
        with open(data_dir / "phase4_trained_models_with_ensemble.pkl", "rb") as f:
            models = pickle.load(f)
        print(f"✅ Loaded Phase 4 models")
    except:
        models = None
        print(f"⚠️  Could not load models")

    # Try to load predictions
    try:
        with open(data_dir / "phase4_save_results.pkl", "rb") as f:
            predictions_data = pickle.load(f)
        print(f"✅ Loaded predictions")
    except:
        predictions_data = None
        print(f"⚠️  Could not load predictions")

except Exception as e:
    print(f"❌ Error loading Phase 4 outputs: {e}")
    print(f"\nLooking for available files in {data_dir}:")
    if data_dir.exists():
        for f in data_dir.glob("*.pkl"):
            print(f"  - {f.name}")
    sys.exit(1)

print()

# ════════════════════════════════════════════════════════════════════════════
# STEP 2: USE PHASE 5a - EVALUATION METRICS
# ════════════════════════════════════════════════════════════════════════════

print("="*80)
print("PHASE 5a: COMPREHENSIVE METRICS")
print("="*80 + "\n")

try:
    from ml_engine.pipelines.evaluation_metrics import ComprehensiveMetricsCalculator

    # Extract data for metrics calculation
    if predictions_data and isinstance(predictions_data, dict):
        y_test = predictions_data.get('y_test')
        y_pred = predictions_data.get('y_pred')
        y_proba = predictions_data.get('y_proba')

        if y_test is not None and y_pred is not None:
            print(f"Calculating metrics...")
            calc = ComprehensiveMetricsCalculator()

            try:
                metrics = calc.evaluate_classification(y_test, y_pred, y_proba)
                print(f"\n✅ Metrics calculated successfully!")
                print(f"\nTop Metrics:")
                print(f"  Accuracy:  {metrics.get('accuracy', 'N/A'):.4f}")
                print(f"  Precision: {metrics.get('precision', 'N/A'):.4f}")
                print(f"  Recall:    {metrics.get('recall', 'N/A'):.4f}")
                print(f"  F1 Score:  {metrics.get('f1_score', 'N/A'):.4f}")
                print(f"  ROC-AUC:   {metrics.get('roc_auc_score', 'N/A'):.4f}")

                # Save metrics
                with open(output_dir / "phase5_metrics.json", "w") as f:
                    # Convert numpy types to native Python types
                    metrics_json = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                                    for k, v in metrics.items()}
                    json.dump(metrics_json, f, indent=2)
                print(f"\n✅ Metrics saved to: {output_dir / 'phase5_metrics.json'}")

            except Exception as e:
                print(f"⚠️  Error calculating metrics: {e}")
        else:
            print(f"⚠️  Could not extract y_test and y_pred from predictions data")
    else:
        print(f"⚠️  Predictions data not found or invalid format")

except ImportError as e:
    print(f"⚠️  Phase 5a module not available: {e}")

print()

# ════════════════════════════════════════════════════════════════════════════
# STEP 3: USE PHASE 5e - VISUALIZATIONS
# ════════════════════════════════════════════════════════════════════════════

print("="*80)
print("PHASE 5e: VISUALIZATIONS")
print("="*80 + "\n")

try:
    from ml_engine.pipelines.visualization_manager import VisualizationManager

    if predictions_data and isinstance(predictions_data, dict):
        y_test = predictions_data.get('y_test')
        y_pred = predictions_data.get('y_pred')
        y_proba = predictions_data.get('y_proba')

        if y_test is not None and y_pred is not None:
            viz = VisualizationManager()

            try:
                # Create confusion matrix
                confusion_path = output_dir / "confusion_matrix.png"
                viz.plot_confusion_matrix(y_test, y_pred, str(confusion_path))
                print(f"✅ Confusion matrix saved to: {confusion_path}")
            except Exception as e:
                print(f"⚠️  Could not create confusion matrix: {e}")

            try:
                # Create ROC curve (if probabilities available)
                if y_proba is not None:
                    roc_path = output_dir / "roc_curve.png"
                    viz.plot_roc_curve(y_test, y_proba, str(roc_path))
                    print(f"✅ ROC curve saved to: {roc_path}")
            except Exception as e:
                print(f"⚠️  Could not create ROC curve: {e}")

            try:
                # Create learning curve
                learning_path = output_dir / "learning_curve.png"
                # Note: This might need the model and training data
                print(f"⚠️  Learning curve requires training data (skipped)")
            except Exception as e:
                print(f"⚠️  Could not create learning curve: {e}")
        else:
            print(f"⚠️  Could not extract predictions from data")
    else:
        print(f"⚠️  Predictions data not found")

except ImportError as e:
    print(f"⚠️  Phase 5e module not available: {e}")

print()

# ════════════════════════════════════════════════════════════════════════════
# STEP 4: USE PHASE 5g - REPORT GENERATION
# ════════════════════════════════════════════════════════════════════════════

print("="*80)
print("PHASE 5g: REPORT GENERATION")
print("="*80 + "\n")

try:
    from ml_engine.pipelines.report_generator import ComprehensiveReportManager

    report = ComprehensiveReportManager("Phase1-4_Model")

    try:
        # Add metrics section (if available)
        if 'metrics' in locals():
            report.add_performance_section(metrics)
            print(f"✅ Added performance metrics to report")
    except Exception as e:
        print(f"⚠️  Could not add metrics section: {e}")

    try:
        # Generate reports
        reports = report.generate_all_reports(str(output_dir))
        print(f"✅ Reports generated successfully!")
        print(f"\nGenerated files:")
        for file in output_dir.glob("*"):
            if file.is_file():
                size = file.stat().st_size / 1024  # KB
                print(f"  - {file.name} ({size:.1f} KB)")
    except Exception as e:
        print(f"⚠️  Error generating reports: {e}")

except ImportError as e:
    print(f"⚠️  Phase 5g module not available: {e}")

print()

# ════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════

print("="*80)
print("✅ PHASE 5 ANALYSIS COMPLETE")
print("="*80)
print(f"\n📊 Results saved to: {output_dir.absolute()}\n")
print("Files generated:")
for file in sorted(output_dir.glob("*")):
    if file.is_file():
        size = file.stat().st_size
        if size > 1024*1024:
            size_str = f"{size/(1024*1024):.1f} MB"
        elif size > 1024:
            size_str = f"{size/1024:.1f} KB"
        else:
            size_str = f"{size} B"
        print(f"  ✅ {file.name} ({size_str})")

print("\n" + "="*80)
print("🎉 PHASE 1-5 PIPELINE COMPLETE!")
print("="*80 + "\n")