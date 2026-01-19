#!/usr/bin/env python3
"""
════════════════════════════════════════════════════════════════════════════
PHASE 5: ADVANCED ANALYSIS & REPORTING (CORRECTED VERSION)
════════════════════════════════════════════════════════════════════════════

This script uses Phase 5 modules to analyze Phase 4 outputs.
CORRECTED to load actual Phase 4 output files

Run this AFTER: kedro run --pipeline complete

Usage:
  python3 phase5_analysis.py
"""

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path

print("\n" + "="*80)
print("PHASE 5: ADVANCED ANALYSIS & REPORTING (CORRECTED)")
print("="*80 + "\n")

# ════════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD PHASE 4 OUTPUTS (ACTUAL FILENAMES)
# ════════════════════════════════════════════════════════════════════════════

print("📂 Step 1: Loading Phase 4 outputs...\n")

data_dir = Path("data/07_model_output")
output_dir = Path("data/08_reporting")
output_dir.mkdir(parents=True, exist_ok=True)

# Try to load actual Phase 4 files
phase4_data = {}
y_test = None
y_pred = None
y_proba = None

try:
    # Load phase4_summary.pkl (most likely to have what we need)
    summary_file = data_dir / "phase4_summary.pkl"
    if summary_file.exists():
        with open(summary_file, "rb") as f:
            phase4_summary = pickle.load(f)
        print(f"✅ Loaded phase4_summary.pkl")
        print(f"   Type: {type(phase4_summary)}")
        if isinstance(phase4_summary, dict):
            print(f"   Keys: {list(phase4_summary.keys())[:5]}...")
        phase4_data['summary'] = phase4_summary
    else:
        print(f"⚠️  phase4_summary.pkl not found")
except Exception as e:
    print(f"⚠️  Error loading summary: {e}")

try:
    # Load predictions from CSV
    predictions_file = data_dir / "phase3_predictions.csv"
    if predictions_file.exists():
        df_pred = pd.read_csv(predictions_file)
        print(f"✅ Loaded phase3_predictions.csv ({len(df_pred)} rows)")
        print(f"   Columns: {list(df_pred.columns)}")

        # Try to extract y_test and y_pred from the dataframe
        if 'y_test' in df_pred.columns:
            y_test = df_pred['y_test'].values
            print(f"   ✅ Found y_test (shape: {y_test.shape})")
        if 'y_pred' in df_pred.columns:
            y_pred = df_pred['y_pred'].values
            print(f"   ✅ Found y_pred (shape: {y_pred.shape})")
        if 'y_proba' in df_pred.columns or 'probability' in df_pred.columns:
            col = 'y_proba' if 'y_proba' in df_pred.columns else 'probability'
            y_proba = df_pred[col].values
            print(f"   ✅ Found {col} (shape: {y_proba.shape})")

        phase4_data['predictions'] = df_pred
    else:
        print(f"⚠️  phase3_predictions.csv not found")
except Exception as e:
    print(f"⚠️  Error loading predictions: {e}")

try:
    # Load phase4_results.csv
    results_file = data_dir / "phase4_results.csv"
    if results_file.exists():
        df_results = pd.read_csv(results_file)
        print(f"✅ Loaded phase4_results.csv ({len(df_results)} rows)")
        print(f"   Columns: {list(df_results.columns)[:5]}...")
        phase4_data['results'] = df_results
    else:
        print(f"⚠️  phase4_results.csv not found")
except Exception as e:
    print(f"⚠️  Error loading results: {e}")

# Check what we loaded
print(f"\n📊 Summary of loaded data:")
print(f"   y_test: {'✅ Loaded' if y_test is not None else '❌ Not found'}")
print(f"   y_pred: {'✅ Loaded' if y_pred is not None else '❌ Not found'}")
print(f"   y_proba: {'✅ Loaded' if y_proba is not None else '❌ Not found'}")
print(f"   phase4_data keys: {list(phase4_data.keys())}\n")

# ════════════════════════════════════════════════════════════════════════════
# STEP 2: USE PHASE 5a - EVALUATION METRICS
# ════════════════════════════════════════════════════════════════════════════

print("="*80)
print("PHASE 5a: COMPREHENSIVE METRICS")
print("="*80 + "\n")

metrics = {}

try:
    from ml_engine.pipelines.evaluation_metrics import ComprehensiveMetricsCalculator

    if y_test is not None and y_pred is not None:
        print(f"Calculating metrics for {len(y_test)} samples...")

        # Ensure arrays are correct shape
        y_test = np.array(y_test).flatten()
        y_pred = np.array(y_pred).flatten()
        if y_proba is not None:
            y_proba = np.array(y_proba).flatten()

        calc = ComprehensiveMetricsCalculator()

        try:
            metrics = calc.evaluate_classification(y_test, y_pred, y_proba)
            print(f"\n✅ Metrics calculated successfully!")
            print(f"\nTop Metrics:")

            metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc_score',
                            'balanced_accuracy', 'specificity', 'sensitivity']
            for name in metric_names:
                val = metrics.get(name, 'N/A')
                if isinstance(val, (float, np.floating)):
                    print(f"  {name.replace('_', ' ').title()}: {val:.4f}")
                else:
                    print(f"  {name.replace('_', ' ').title()}: {val}")

            # Save metrics
            with open(output_dir / "phase5_metrics.json", "w") as f:
                # Convert numpy types to native Python types
                metrics_json = {}
                for k, v in metrics.items():
                    if isinstance(v, (np.floating, np.integer)):
                        metrics_json[k] = float(v)
                    elif isinstance(v, np.ndarray):
                        metrics_json[k] = v.tolist()
                    else:
                        metrics_json[k] = v
                json.dump(metrics_json, f, indent=2)
            print(f"\n✅ Metrics saved to: data/08_reporting/phase5_metrics.json")

        except Exception as e:
            print(f"⚠️  Error calculating metrics: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"⚠️  Cannot calculate metrics: y_test={y_test is not None}, y_pred={y_pred is not None}")

except ImportError as e:
    print(f"⚠️  Phase 5a module not available: {e}")
except Exception as e:
    print(f"⚠️  Unexpected error: {e}")
    import traceback
    traceback.print_exc()

print()

# ════════════════════════════════════════════════════════════════════════════
# STEP 3: USE PHASE 5e - VISUALIZATIONS
# ════════════════════════════════════════════════════════════════════════════

print("="*80)
print("PHASE 5e: VISUALIZATIONS")
print("="*80 + "\n")

try:
    from ml_engine.pipelines.visualization_manager import VisualizationManager

    if y_test is not None and y_pred is not None:
        viz = VisualizationManager()

        # Create confusion matrix
        try:
            confusion_path = output_dir / "confusion_matrix.png"
            viz.plot_confusion_matrix(y_test, y_pred, str(confusion_path))
            print(f"✅ Confusion matrix saved to: data/08_reporting/confusion_matrix.png")
        except Exception as e:
            print(f"⚠️  Could not create confusion matrix: {e}")

        # Create ROC curve (if probabilities available)
        if y_proba is not None:
            try:
                roc_path = output_dir / "roc_curve.png"
                viz.plot_roc_curve(y_test, y_proba, str(roc_path))
                print(f"✅ ROC curve saved to: data/08_reporting/roc_curve.png")
            except Exception as e:
                print(f"⚠️  Could not create ROC curve: {e}")
    else:
        print(f"⚠️  Cannot create visualizations: missing y_test or y_pred")

except ImportError as e:
    print(f"⚠️  Phase 5e module not available: {e}")
except Exception as e:
    print(f"⚠️  Unexpected error: {e}")
    import traceback
    traceback.print_exc()

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

    # Add metrics section
    try:
        if metrics:
            report.add_performance_section(metrics)
            print(f"✅ Added performance metrics to report")
    except Exception as e:
        print(f"⚠️  Could not add metrics section: {e}")

    # Generate reports
    try:
        reports = report.generate_all_reports(str(output_dir))
        print(f"✅ Reports generated successfully!")
    except Exception as e:
        print(f"⚠️  Error generating reports: {e}")

except ImportError as e:
    print(f"⚠️  Phase 5g module not available: {e}")
except Exception as e:
    print(f"⚠️  Unexpected error: {e}")

print()

# ════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════════

print("="*80)
print("✅ PHASE 5 ANALYSIS COMPLETE")
print("="*80)
print(f"\n📊 Results saved to: {output_dir.absolute()}\n")

if output_dir.exists():
    files = list(output_dir.glob("*"))
    if files:
        print("Files generated:")
        for file in sorted(files):
            if file.is_file():
                size = file.stat().st_size
                if size > 1024*1024:
                    size_str = f"{size/(1024*1024):.1f} MB"
                elif size > 1024:
                    size_str = f"{size/1024:.1f} KB"
                else:
                    size_str = f"{size} B"
                print(f"  ✅ {file.name} ({size_str})")
    else:
        print("No files generated yet")
else:
    print(f"Output directory does not exist")

print("\n" + "="*80)
print("🎉 PHASE 1-5 PIPELINE COMPLETE!")
print("="*80 + "\n")

# Print summary statistics
if metrics:
    print("📊 FINAL METRICS SUMMARY:")
    print(f"  Accuracy:  {metrics.get('accuracy', 'N/A')}")
    print(f"  F1 Score:  {metrics.get('f1_score', 'N/A')}")
    print(f"  ROC-AUC:   {metrics.get('roc_auc_score', 'N/A')}")
    print()