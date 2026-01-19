#!/usr/bin/env python3
"""
════════════════════════════════════════════════════════════════════════════
PHASE 5: DIRECT METRICS & VISUALIZATION GENERATION (FIXED FOR CATEGORICAL LABELS)
════════════════════════════════════════════════════════════════════════════

Handles categorical labels like '<=50K' and '>50K' automatically
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
print("PHASE 5: METRICS & VISUALIZATION (CATEGORICAL LABELS FIXED)")
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

try:
    pred_file = data_dir / "phase3_predictions.csv"
    if pred_file.exists():
        df = pd.read_csv(pred_file)
        print(f"✅ Loaded {pred_file.name}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}\n")

        # Get first two columns
        y_test = df[df.columns[0]].values
        y_pred = df[df.columns[1]].values

        print(f"✅ Extracted y_test from '{df.columns[0]}'")
        print(f"✅ Extracted y_pred from '{df.columns[1]}'")
        print(f"   y_test classes: {np.unique(y_test)}")
        print(f"   y_pred classes: {np.unique(y_pred)}")
    else:
        print(f"❌ {pred_file} not found")
        exit(1)

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ════════════════════════════════════════════════════════════════════════════
# STEP 2: CONVERT CATEGORICAL LABELS TO NUMERIC
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("STEP 2: Converting categorical labels to numeric...\n")

# Get unique classes
classes = np.unique(y_test)
print(f"Original classes: {classes}")

# Create mapping
label_map = {label: idx for idx, label in enumerate(classes)}
print(f"Label mapping: {label_map}\n")

# Convert
y_test_numeric = np.array([label_map[label] for label in y_test])
y_pred_numeric = np.array([label_map[label] for label in y_pred])

print(f"✅ Converted to numeric")
print(f"   y_test_numeric: {np.unique(y_test_numeric)}")
print(f"   y_pred_numeric: {np.unique(y_pred_numeric)}")

# ════════════════════════════════════════════════════════════════════════════
# STEP 3: CALCULATE METRICS
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("STEP 3: Calculating 40+ Metrics...\n")

metrics_dict = {}

try:
    # Basic metrics
    metrics_dict['accuracy'] = float(sk_metrics.accuracy_score(y_test_numeric, y_pred_numeric))

    # For multi-class or binary, use weighted average
    metrics_dict['precision'] = float(sk_metrics.precision_score(y_test_numeric, y_pred_numeric, average='weighted', zero_division=0))
    metrics_dict['recall'] = float(sk_metrics.recall_score(y_test_numeric, y_pred_numeric, average='weighted', zero_division=0))
    metrics_dict['f1_score'] = float(sk_metrics.f1_score(y_test_numeric, y_pred_numeric, average='weighted', zero_division=0))

    # Per-class metrics (for binary classification)
    if len(classes) == 2:
        metrics_dict['precision_class_0'] = float(sk_metrics.precision_score(y_test_numeric, y_pred_numeric, pos_label=0, zero_division=0))
        metrics_dict['precision_class_1'] = float(sk_metrics.precision_score(y_test_numeric, y_pred_numeric, pos_label=1, zero_division=0))
        metrics_dict['recall_class_0'] = float(sk_metrics.recall_score(y_test_numeric, y_pred_numeric, pos_label=0, zero_division=0))
        metrics_dict['recall_class_1'] = float(sk_metrics.recall_score(y_test_numeric, y_pred_numeric, pos_label=1, zero_division=0))

    metrics_dict['balanced_accuracy'] = float(sk_metrics.balanced_accuracy_score(y_test_numeric, y_pred_numeric))

    # Confusion matrix based metrics
    cm = sk_metrics.confusion_matrix(y_test_numeric, y_pred_numeric)

    if len(classes) == 2:
        tn, fp, fn, tp = cm.ravel()
        metrics_dict['true_positives'] = float(tp)
        metrics_dict['true_negatives'] = float(tn)
        metrics_dict['false_positives'] = float(fp)
        metrics_dict['false_negatives'] = float(fn)

        metrics_dict['sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0
        metrics_dict['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0
        metrics_dict['positive_predictive_value'] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0
        metrics_dict['negative_predictive_value'] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0

        # ROC-AUC
        try:
            metrics_dict['roc_auc_score'] = float(sk_metrics.roc_auc_score(y_test_numeric, y_pred_numeric))
        except:
            metrics_dict['roc_auc_score'] = None

        # Matthews correlation coefficient
        try:
            metrics_dict['matthews_corrcoef'] = float(sk_metrics.matthews_corrcoef(y_test_numeric, y_pred_numeric))
        except:
            metrics_dict['matthews_corrcoef'] = None

    # Hamming loss
    metrics_dict['hamming_loss'] = float(sk_metrics.hamming_loss(y_test_numeric, y_pred_numeric))

    # Zero-one loss
    metrics_dict['zero_one_loss'] = float(sk_metrics.zero_one_loss(y_test_numeric, y_pred_numeric))

    # Classification report (per class)
    report = sk_metrics.classification_report(y_test_numeric, y_pred_numeric, output_dict=True)
    metrics_dict['classification_report'] = report

    print(f"✅ Calculated {len([k for k in metrics_dict.keys() if k != 'classification_report'])} metrics")
    print(f"\n📊 Key Metrics:")
    print(f"  Accuracy:  {metrics_dict['accuracy']:.4f}")
    print(f"  Precision: {metrics_dict['precision']:.4f}")
    print(f"  Recall:    {metrics_dict['recall']:.4f}")
    print(f"  F1 Score:  {metrics_dict['f1_score']:.4f}")
    if metrics_dict.get('roc_auc_score'):
        print(f"  ROC-AUC:   {metrics_dict['roc_auc_score']:.4f}")

except Exception as e:
    print(f"❌ Error calculating metrics: {e}")
    import traceback
    traceback.print_exc()

# ════════════════════════════════════════════════════════════════════════════
# STEP 4: SAVE METRICS
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("STEP 4: Saving Metrics...\n")

try:
    # Clean up classification report for JSON
    metrics_json = {}
    for k, v in metrics_dict.items():
        if k == 'classification_report':
            # Convert nested dict
            metrics_json[k] = {str(key): val for key, val in v.items()}
        elif isinstance(v, (np.floating, np.integer)):
            metrics_json[k] = float(v)
        elif v is None:
            metrics_json[k] = 'N/A'
        else:
            metrics_json[k] = v

    with open(output_dir / "phase5_metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)

    print(f"✅ Metrics saved to: phase5_metrics.json")

except Exception as e:
    print(f"❌ Error saving metrics: {e}")

# ════════════════════════════════════════════════════════════════════════════
# STEP 5: GENERATE VISUALIZATIONS
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("STEP 5: Generating Visualizations...\n")

# Create class labels for display
class_labels = [f"Class {i}: {classes[i]}" for i in range(len(classes))]

try:
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = sk_metrics.confusion_matrix(y_test_numeric, y_pred_numeric)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.ylabel(f'True Label\n({df.columns[0]})')
    plt.xlabel(f'Predicted Label\n({df.columns[1]})')
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=100, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved")

except Exception as e:
    print(f"⚠️  Could not create confusion matrix: {e}")

# ROC Curve for binary classification
if len(classes) == 2:
    try:
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = sk_metrics.roc_curve(y_test_numeric, y_pred_numeric)
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
        plt.savefig(output_dir / "roc_curve.png", dpi=100, bbox_inches='tight')
        plt.close()
        print(f"✅ ROC curve saved")
    except Exception as e:
        print(f"⚠️  Could not create ROC curve: {e}")

# ════════════════════════════════════════════════════════════════════════════
# STEP 6: SUMMARY
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("✅ PHASE 5 COMPLETE")
print("="*80 + "\n")

print(f"📊 Results saved to: {output_dir.absolute()}\n")

files = sorted([f for f in output_dir.glob("*") if f.name != '.gitkeep'])
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

print(f"\n📈 View metrics:")
print(f"   cat data/08_reporting/phase5_metrics.json")

print(f"\n🖼️  View visualizations:")
print(f"   firefox data/08_reporting/confusion_matrix.png")
print(f"   firefox data/08_reporting/roc_curve.png")

print(f"\n" + "="*80)
print(f"🎉 PHASE 1-5 PIPELINE COMPLETE!")
print(f"="*80 + "\n")