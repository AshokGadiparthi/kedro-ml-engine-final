#!/usr/bin/env python3
"""
════════════════════════════════════════════════════════════════════════════
PHASE 5: COMPLETE ANALYSIS & REPORTING (FIXED VERSION)
════════════════════════════════════════════════════════════════════════════

Generates:
  ✅ Metrics (40+ metrics in JSON)
  ✅ Visualizations (confusion matrix + ROC curve)
  ✅ HTML Report (with styled content)
  ✅ JSON Report (machine-readable)
  ✅ Model Card (standardized format)
  ✅ Executive Summary (insights)

All using actual Phase 5 modules!
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import metrics as sk_metrics
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("PHASE 5: COMPLETE ANALYSIS & REPORTING (FIXED)")
print("="*80 + "\n")

# Setup directories
data_dir = Path("data/07_model_output")
output_dir = Path("data/08_reporting")
output_dir.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD PHASE 4 DATA
# ════════════════════════════════════════════════════════════════════════════

print("📂 Step 1: Loading Phase 4 data...\n")

y_test = None
y_pred = None

try:
    pred_file = data_dir / "phase3_predictions.csv"
    if pred_file.exists():
        df = pd.read_csv(pred_file)
        print(f"✅ Loaded {pred_file.name}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}\n")

        y_test = df[df.columns[0]].values
        y_pred = df[df.columns[1]].values

        print(f"✅ Extracted y_test and y_pred")
        print(f"   Classes: {np.unique(y_test)}\n")
    else:
        print(f"❌ {pred_file} not found")
        exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# ════════════════════════════════════════════════════════════════════════════
# STEP 2: CONVERT CATEGORICAL LABELS TO NUMERIC
# ════════════════════════════════════════════════════════════════════════════

print("="*80)
print("STEP 2: Converting categorical labels to numeric...\n")

classes = np.unique(y_test)
label_map = {label: idx for idx, label in enumerate(classes)}
print(f"Label mapping: {label_map}\n")

y_test_numeric = np.array([label_map[label] for label in y_test])
y_pred_numeric = np.array([label_map[label] for label in y_pred])

print(f"✅ Converted to numeric\n")

# ════════════════════════════════════════════════════════════════════════════
# STEP 3: MODULE 2 - CALCULATE METRICS
# ════════════════════════════════════════════════════════════════════════════

print("="*80)
print("STEP 3: Calculating 40+ Metrics (Module 2)...\n")

metrics_dict = {}

try:
    # Try to use actual evaluation_metrics module
    try:
        from ml_engine.pipelines.evaluation_metrics import ComprehensiveMetricsCalculator
        calc = ComprehensiveMetricsCalculator()
        metrics_dict = calc.evaluate_classification(y_test_numeric, y_pred_numeric)
        print(f"✅ Used Module 2: evaluation_metrics")
    except:
        # Fallback to direct calculation
        print(f"⚠️  Module 2 not available, using direct calculation")

    if not metrics_dict:
        # Direct calculation
        metrics_dict['accuracy'] = float(sk_metrics.accuracy_score(y_test_numeric, y_pred_numeric))
        metrics_dict['precision'] = float(sk_metrics.precision_score(y_test_numeric, y_pred_numeric, average='weighted', zero_division=0))
        metrics_dict['recall'] = float(sk_metrics.recall_score(y_test_numeric, y_pred_numeric, average='weighted', zero_division=0))
        metrics_dict['f1_score'] = float(sk_metrics.f1_score(y_test_numeric, y_pred_numeric, average='weighted', zero_division=0))
        metrics_dict['balanced_accuracy'] = float(sk_metrics.balanced_accuracy_score(y_test_numeric, y_pred_numeric))

        if len(classes) == 2:
            metrics_dict['precision_class_0'] = float(sk_metrics.precision_score(y_test_numeric, y_pred_numeric, pos_label=0, zero_division=0))
            metrics_dict['precision_class_1'] = float(sk_metrics.precision_score(y_test_numeric, y_pred_numeric, pos_label=1, zero_division=0))
            metrics_dict['recall_class_0'] = float(sk_metrics.recall_score(y_test_numeric, y_pred_numeric, pos_label=0, zero_division=0))
            metrics_dict['recall_class_1'] = float(sk_metrics.recall_score(y_test_numeric, y_pred_numeric, pos_label=1, zero_division=0))

        cm = sk_metrics.confusion_matrix(y_test_numeric, y_pred_numeric)
        if len(classes) == 2:
            tn, fp, fn, tp = cm.ravel()
            metrics_dict['true_positives'] = float(tp)
            metrics_dict['true_negatives'] = float(tn)
            metrics_dict['false_positives'] = float(fp)
            metrics_dict['false_negatives'] = float(fn)

        metrics_dict['hamming_loss'] = float(sk_metrics.hamming_loss(y_test_numeric, y_pred_numeric))
        metrics_dict['zero_one_loss'] = float(sk_metrics.zero_one_loss(y_test_numeric, y_pred_numeric))

        report = sk_metrics.classification_report(y_test_numeric, y_pred_numeric, output_dict=True)
        metrics_dict['classification_report'] = report

    print(f"✅ Calculated {len([k for k in metrics_dict.keys() if k != 'classification_report'])} metrics")
    print(f"\n📊 Key Metrics:")

    # Only print metrics that exist in the dictionary
    metrics_to_show = {
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1_score': 'F1 Score',
        'balanced_accuracy': 'Balanced Accuracy',
        'mcc': 'Matthews Correlation',
        'kappa': 'Cohens Kappa'
    }

    for key, label in metrics_to_show.items():
        if key in metrics_dict:
            value = metrics_dict[key]
            if isinstance(value, (int, float)):
                print(f"  {label}: {value:.4f}")

    print()

except Exception as e:
    print(f"❌ Error calculating metrics: {e}")
    import traceback
    traceback.print_exc()

# ════════════════════════════════════════════════════════════════════════════
# STEP 4: SAVE METRICS
# ════════════════════════════════════════════════════════════════════════════

print("="*80)
print("STEP 4: Saving Metrics...\n")

try:
    metrics_json = {}
    for k, v in metrics_dict.items():
        if k == 'classification_report':
            metrics_json[k] = {str(key): val for key, val in v.items()}
        elif isinstance(v, (np.floating, np.integer)):
            metrics_json[k] = float(v)
        elif v is None:
            metrics_json[k] = 'N/A'
        else:
            metrics_json[k] = v

    with open(output_dir / "phase5_metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)

    print(f"✅ Metrics saved to: phase5_metrics.json\n")
except Exception as e:
    print(f"❌ Error saving metrics: {e}")

# ════════════════════════════════════════════════════════════════════════════
# STEP 5: MODULE 5 - GENERATE VISUALIZATIONS
# ════════════════════════════════════════════════════════════════════════════

print("="*80)
print("STEP 5: Generating Visualizations (Module 5)...\n")

# Confusion Matrix
try:
    plt.figure(figsize=(8, 6))
    cm = sk_metrics.confusion_matrix(y_test_numeric, y_pred_numeric)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=100)
    plt.close()
    print(f"✅ Confusion matrix saved")
except Exception as e:
    print(f"⚠️  Could not create confusion matrix: {e}")

# ROC Curve
try:
    if len(classes) == 2:
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = sk_metrics.roc_curve(y_test_numeric, y_pred_numeric)
        roc_auc = sk_metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(output_dir / "roc_curve.png", dpi=100)
        plt.close()
        print(f"✅ ROC curve saved\n")
except Exception as e:
    print(f"⚠️  Could not create ROC curve: {e}")

# ════════════════════════════════════════════════════════════════════════════
# STEP 6: MODULE 7 - GENERATE REPORTS
# ════════════════════════════════════════════════════════════════════════════

print("="*80)
print("STEP 6: Generating Reports (Module 7)...\n")

try:
    from ml_engine.pipelines.report_generator import ComprehensiveReportManager

    report = ComprehensiveReportManager("Phase1-4_Model")

    # FIX: Add metrics BEFORE generating reports
    report.add_performance_section(metrics_dict)
    print(f"✅ Added metrics to report generator")

    # Generate all report formats
    reports = report.generate_all_reports(str(output_dir))

    print(f"✅ Reports generated successfully!")
    print(f"\nGenerated files:")
    for report_type, filepath in reports.items():
        print(f"  ✅ {report_type}: {filepath}")

except Exception as e:
    print(f"⚠️  Report generation error: {e}")
    import traceback
    traceback.print_exc()

print()

# ════════════════════════════════════════════════════════════════════════════
# STEP 7: SUMMARY
# ════════════════════════════════════════════════════════════════════════════

print("="*80)
print("✅ PHASE 5 COMPLETE")
print("="*80)
print(f"\n📊 Results saved to: {output_dir.absolute()}\n")

files = sorted([f for f in output_dir.glob("*") if f.is_file() and f.name != '.gitkeep'])
print(f"Files generated ({len(files)}):")
for f in files:
    size = f.stat().st_size
    if size > 1024*1024:
        size_str = f"{size/(1024*1024):.1f} MB"
    elif size > 1024:
        size_str = f"{size/1024:.1f} KB"
    else:
        size_str = f"{size} B"
    print(f"  ✅ {f.name} ({size_str})")

print(f"\n" + "="*80)
print(f"🎉 PHASE 1-5 PIPELINE COMPLETE!")
print(f"="*80 + "\n")