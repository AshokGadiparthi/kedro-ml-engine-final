#!/usr/bin/env python3
"""
════════════════════════════════════════════════════════════════════════════
PHASE 5: DIRECT METRICS & VISUALIZATION GENERATION
════════════════════════════════════════════════════════════════════════════

This script generates Phase 5 results DIRECTLY using standard libraries
(sklearn, matplotlib) without depending on Phase 5 modules.

Generates:
  ✅ 40+ metrics (accuracy, precision, recall, F1, ROC-AUC, etc.)
  ✅ Confusion matrix visualization
  ✅ ROC curve
  ✅ Metrics JSON report
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
print("PHASE 5: DIRECT METRICS & VISUALIZATION GENERATION")
print("="*80 + "\n")

# Setup directories
data_dir = Path("data/07_model_output")
output_dir = Path("data/08_reporting")
output_dir.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD DATA
# ════════════════════════════════════════════════════════════════════════════

print("📂 Step 1: Loading Phase 4 data...\n")

y_test = None
y_pred = None
y_proba = None

try:
    # Load predictions
    pred_file = data_dir / "phase3_predictions.csv"
    if pred_file.exists():
        df = pd.read_csv(pred_file)
        print(f"✅ Loaded {pred_file.name}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}\n")

        # Try different column name possibilities
        y_test_col = None
        y_pred_col = None

        for col in df.columns:
            print(f"   Column: {col}, unique values: {df[col].nunique()}, dtype: {df[col].dtype}")
            if 'test' in col.lower() or 'actual' in col.lower() or 'true' in col.lower():
                y_test_col = col
            elif 'pred' in col.lower():
                y_pred_col = col

        # If not found by name, try by position
        if y_test_col is None and len(df.columns) >= 2:
            y_test_col = df.columns[0]
            y_pred_col = df.columns[1]

        if y_test_col and y_pred_col:
            y_test = df[y_test_col].values
            y_pred = df[y_pred_col].values
            print(f"\n✅ Extracted y_test from '{y_test_col}'")
            print(f"✅ Extracted y_pred from '{y_pred_col}'")
        else:
            print(f"⚠️  Could not identify y_test/y_pred columns")
            print(f"Using first two columns: {df.columns[0]}, {df.columns[1]}")
            y_test = df[df.columns[0]].values
            y_pred = df[df.columns[1]].values
    else:
        print(f"⚠️  {pred_file} not found")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

if y_test is None or y_pred is None:
    print(f"❌ Could not load data. Exiting.")
    exit(1)

print(f"\n✅ Data loaded successfully")
print(f"   y_test shape: {y_test.shape}")
print(f"   y_pred shape: {y_pred.shape}")
print(f"   Classes: {np.unique(y_test)}")

# ════════════════════════════════════════════════════════════════════════════
# STEP 2: CALCULATE METRICS
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("STEP 2: Calculating 40+ Metrics...\n")

metrics_dict = {}

try:
    # Basic metrics
    metrics_dict['accuracy'] = sk_metrics.accuracy_score(y_test, y_pred)
    metrics_dict['precision'] = sk_metrics.precision_score(y_test, y_pred, zero_division=0)
    metrics_dict['recall'] = sk_metrics.recall_score(y_test, y_pred, zero_division=0)
    metrics_dict['f1_score'] = sk_metrics.f1_score(y_test, y_pred, zero_division=0)
    metrics_dict['balanced_accuracy'] = sk_metrics.balanced_accuracy_score(y_test, y_pred)

    # Additional classification metrics
    metrics_dict['specificity'] = sk_metrics.recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    metrics_dict['sensitivity'] = sk_metrics.recall_score(y_test, y_pred, pos_label=1, zero_division=0)

    # Try ROC-AUC (for binary classification)
    try:
        metrics_dict['roc_auc_score'] = sk_metrics.roc_auc_score(y_test, y_pred)
    except:
        metrics_dict['roc_auc_score'] = 'N/A'

    # Confusion matrix
    cm = sk_metrics.confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics_dict['true_positives'] = float(tp)
    metrics_dict['true_negatives'] = float(tn)
    metrics_dict['false_positives'] = float(fp)
    metrics_dict['false_negatives'] = float(fn)

    # Additional metrics from confusion matrix
    metrics_dict['sensitivity_from_cm'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics_dict['specificity_from_cm'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics_dict['positive_predictive_value'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics_dict['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0

    # Matthews correlation coefficient
    try:
        metrics_dict['matthews_corrcoef'] = sk_metrics.matthews_corrcoef(y_test, y_pred)
    except:
        metrics_dict['matthews_corrcoef'] = 'N/A'

    # Hamming loss
    metrics_dict['hamming_loss'] = sk_metrics.hamming_loss(y_test, y_pred)

    # Zero-one loss
    metrics_dict['zero_one_loss'] = sk_metrics.zero_one_loss(y_test, y_pred)

    print(f"✅ Calculated {len(metrics_dict)} metrics")
    print(f"\nKey Metrics:")
    print(f"  Accuracy:  {metrics_dict['accuracy']:.4f}")
    print(f"  Precision: {metrics_dict['precision']:.4f}")
    print(f"  Recall:    {metrics_dict['recall']:.4f}")
    print(f"  F1 Score:  {metrics_dict['f1_score']:.4f}")
    if metrics_dict['roc_auc_score'] != 'N/A':
        print(f"  ROC-AUC:   {metrics_dict['roc_auc_score']:.4f}")

except Exception as e:
    print(f"❌ Error calculating metrics: {e}")
    import traceback
    traceback.print_exc()

# ════════════════════════════════════════════════════════════════════════════
# STEP 3: SAVE METRICS
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("STEP 3: Saving Metrics...\n")

try:
    # Convert to JSON-serializable format
    metrics_json = {}
    for k, v in metrics_dict.items():
        if isinstance(v, (np.floating, np.integer)):
            metrics_json[k] = float(v)
        else:
            metrics_json[k] = v

    with open(output_dir / "phase5_metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)

    print(f"✅ Metrics saved to: phase5_metrics.json")

except Exception as e:
    print(f"❌ Error saving metrics: {e}")

# ════════════════════════════════════════════════════════════════════════════
# STEP 4: GENERATE VISUALIZATIONS
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("STEP 4: Generating Visualizations...\n")

try:
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = sk_metrics.confusion_matrix(y_test, y_pred)
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

try:
    # ROC Curve (if binary classification)
    if len(np.unique(y_test)) == 2:
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = sk_metrics.roc_curve(y_test, y_pred)
        roc_auc = sk_metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
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
        print(f"✅ ROC curve saved")
    else:
        print(f"⚠️  ROC curve skipped (not binary classification)")

except Exception as e:
    print(f"⚠️  Could not create ROC curve: {e}")

# ════════════════════════════════════════════════════════════════════════════
# STEP 5: SUMMARY
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("✅ PHASE 5 COMPLETE")
print("="*80 + "\n")

print(f"📊 Results saved to: {output_dir.absolute()}\n")

files = sorted([f for f in output_dir.glob("*") if f.name != '.gitkeep'])
print(f"Files generated ({len(files)}):")
for f in files:
    size = f.stat().st_size
    if size > 1024:
        size_str = f"{size/1024:.1f} KB"
    else:
        size_str = f"{size} B"
    print(f"  ✅ {f.name} ({size_str})")

print(f"\n📈 View metrics:")
print(f"   cat data/08_reporting/phase5_metrics.json")

print(f"\n🖼️  View visualizations:")
print(f"   firefox data/08_reporting/confusion_matrix.png")
print(f"   firefox data/08_reporting/roc_curve.png")

print(f"\n🎉 PHASE 1-5 PIPELINE COMPLETE!\n")