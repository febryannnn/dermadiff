"""
DermaDiff - Evaluation Script
Evaluate trained PanDerm classifiers on test set and compare all experiments.

Usage (Google Colab):
    1. Run panderm_classifier.py first
    2. Run this script: !python evaluation.py
    3. Outputs per-class metrics + cross-experiment comparison

Can also be run standalone if checkpoints already exist on Drive.
"""

import os
import json

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = '/content/drive/MyDrive/DermaDiff'
SHARED = os.path.join(PROJECT_ROOT, 'shared')
NOTEBOOK_DIR = os.path.join(PROJECT_ROOT, 'notebooks/exp_c1')

with open(os.path.join(SHARED, 'config/shared_config.json')) as f:
    CONFIG = json.load(f)
with open(os.path.join(SHARED, 'config/label_mapping.json')) as f:
    label_map = json.load(f)
inv_map = {v: k for k, v in label_map.items()}

PANDERM_DIR = '/content/PanDerm'
TARGET_CLASSES = ['mel', 'bcc', 'akiec', 'df', 'vasc']

# Paths for C1 experiments
C1_1X_OUTPUT = '/content/exp_c1_1x_output'
C1_2X_OUTPUT = '/content/exp_c1_2x_output'
C1_1X_DRIVE = os.path.join(NOTEBOOK_DIR, 'outputs/exp_c1_1x')
C1_2X_DRIVE = os.path.join(NOTEBOOK_DIR, 'outputs/exp_c1_2x')
C1_1X_IMAGES = '/content/exp_c1_1x_images'
C1_2X_IMAGES = '/content/exp_c1_2x_images'

TEMP_DIR = os.path.join(NOTEBOOK_DIR, 'temp')
c1_1x_csv = os.path.join(TEMP_DIR, 'ham10000_exp_c1_1x.csv')
c1_2x_csv = os.path.join(TEMP_DIR, 'ham10000_exp_c1_2x.csv')

# ============================================================
# 2. EVALUATE ON TEST SET
# ============================================================

def run_evaluation(checkpoint_path, csv_path, images_dir, output_dir, name):
    """Run PanDerm evaluation on test set using --eval flag."""
    import torch
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    batch_size = 16 if vram < 20 else CONFIG['batch_size']

    print(f"\nEvaluating {name} on test set...")
    cmd = f"""cd {PANDERM_DIR}/classification && \
CUDA_VISIBLE_DEVICES=0 python3 run_class_finetuning.py \
    --model {CONFIG['model_name']} \
    --nb_classes {CONFIG['nb_classes']} \
    --batch_size {batch_size} \
    --sin_pos_emb \
    --imagenet_default_mean_and_std \
    --eval \
    --resume {checkpoint_path} \
    --csv_path {csv_path} \
    --root_path "{images_dir}/" \
    --output_dir {output_dir}"""

    exit_code = os.system(cmd)
    if exit_code != 0:
        print(f"  WARNING: Evaluation failed for {name}")
    return exit_code == 0


# ============================================================
# 3. COMPUTE METRICS
# ============================================================

def compute_metrics(csv_path, label_name='C1'):
    """Compute all metrics from PanDerm's test output CSV."""
    if not os.path.exists(csv_path):
        print(f"  {label_name}: test.csv not found at {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    # Handle both column naming conventions
    if 'label' in df.columns:
        y_true = df['label'].values
    elif 'true_label' in df.columns:
        y_true = df['true_label'].values
    else:
        print(f"  {label_name}: Cannot find label column. Available: {list(df.columns)}")
        return None

    if 'prediction' in df.columns:
        y_pred = df['prediction'].values
    elif 'predicted_label' in df.columns:
        y_pred = df['predicted_label'].values
    else:
        print(f"  {label_name}: Cannot find prediction column. Available: {list(df.columns)}")
        return None

    acc = accuracy_score(y_true, y_pred)
    w_f1 = f1_score(y_true, y_pred, average='weighted')
    m_f1 = f1_score(y_true, y_pred, average='macro')
    w_rec = recall_score(y_true, y_pred, average='weighted')
    m_rec = recall_score(y_true, y_pred, average='macro')

    class_names = [inv_map[i] for i in sorted(inv_map.keys())]
    rec_per = recall_score(y_true, y_pred, average=None)
    prec_per = precision_score(y_true, y_pred, average=None)
    f1_per = f1_score(y_true, y_pred, average=None)

    print(f"\n{'='*60}")
    print(f"  {label_name} - TEST SET RESULTS")
    print(f"{'='*60}")
    print(f"  Accuracy:        {acc:.4f}")
    print(f"  Weighted F1:     {w_f1:.4f}")
    print(f"  Macro F1:        {m_f1:.4f}")
    print(f"  Weighted Recall: {w_rec:.4f}")
    print(f"  Macro Recall:    {m_rec:.4f}")

    print(f"\n  PER-CLASS METRICS:")
    print(f"  {'Class':>6s}  {'Prec':>6s}  {'Recall':>6s}  {'F1':>6s}  {'N':>5s}")
    print(f"  {'-'*38}")

    for i, cls in enumerate(class_names):
        n = (y_true == i).sum()
        print(f"  {cls:>6s}  {prec_per[i]:>6.3f}  {rec_per[i]:>6.3f}  {f1_per[i]:>6.3f}  {n:>5d}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n  CONFUSION MATRIX:")
    header = "  " + f"{'':>6s}" + "".join(f"{cls:>6s}" for cls in class_names)
    print(header)
    for i, cls in enumerate(class_names):
        row = f"  {cls:>6s}" + "".join(f"{cm[i][j]:>6d}" for j in range(len(class_names)))
        print(row)

    return {
        'acc': acc, 'w_f1': w_f1, 'm_f1': m_f1,
        'w_rec': w_rec, 'm_rec': m_rec,
        'recall': dict(zip(class_names, rec_per.tolist())),
        'precision': dict(zip(class_names, prec_per.tolist())),
        'f1': dict(zip(class_names, f1_per.tolist())),
    }


# ============================================================
# 4. CROSS-EXPERIMENT COMPARISON
# ============================================================

def compare_all_experiments(r_1x, r_2x):
    """Compare all experiments side by side."""

    # Previous experiment results (hardcoded from earlier runs)
    exp_a = {'acc': 0.8756, 'w_f1': 0.8785, 'm_f1': 0.8114,
             'recall': {'akiec': 0.735, 'bcc': 0.909, 'bkl': 0.842,
                        'df': 0.588, 'mel': 0.713, 'nv': 0.917, 'vasc': 0.864}}
    exp_b = {'acc': 0.8829, 'w_f1': 0.8876, 'm_f1': 0.8359,
             'recall': {'akiec': 0.776, 'bcc': 0.922, 'bkl': 0.842,
                        'df': 0.647, 'mel': 0.820, 'nv': 0.912, 'vasc': 0.864}}
    c2_5x = {'acc': 0.8829, 'w_f1': 0.8868, 'm_f1': 0.8340,
             'recall': {'akiec': 0.796, 'bcc': 0.883, 'bkl': 0.836,
                        'df': 0.706, 'mel': 0.802, 'nv': 0.912, 'vasc': 0.864}}
    c3_1x = {'acc': 0.8842, 'w_f1': 0.8891, 'm_f1': 0.8409,
             'recall': {'akiec': 0.837, 'bcc': 0.870, 'bkl': 0.836,
                        'df': 0.706, 'mel': 0.790, 'nv': 0.915, 'vasc': 0.864}}
    c3_2x = {'acc': 0.8922, 'w_f1': 0.8920, 'm_f1': 0.8279,
             'recall': {'akiec': 0.755, 'bcc': 0.883, 'bkl': 0.842,
                        'df': 0.588, 'mel': 0.677, 'nv': 0.949, 'vasc': 0.864}}

    print(f"\n{'='*85}")
    print("  FULL COMPARISON: All Experiments")
    print("=" * 85)

    all_exps = [
        ('Exp A', exp_a), ('Exp B', exp_b), ('C2-5x', c2_5x),
        ('C3-1x', c3_1x), ('C3-2x', c3_2x),
    ]
    if r_1x:
        all_exps.append(('C1-1x', r_1x))
    if r_2x:
        all_exps.append(('C1-2x', r_2x))

    header = f"{'Metric':>15s}" + ''.join(f" | {name:>8s}" for name, _ in all_exps)
    print(header)
    print("-" * len(header))

    for metric_name, metric_key in [('Accuracy', 'acc'), ('Weighted F1', 'w_f1'), ('Macro F1', 'm_f1')]:
        row = f"{metric_name:>15s}"
        for _, exp in all_exps:
            val = exp.get(metric_key, 0)
            row += f" | {val:>8.4f}"
        print(row)

    print(f"\n{'':>15s} --- PER-CLASS RECALL ---")
    for cls in ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']:
        row = f"{cls:>15s}"
        vals = []
        for _, exp in all_exps:
            v = exp.get('recall', {}).get(cls, 0)
            row += f" | {v:>8.3f}"
            vals.append(v)
        best = max(vals)
        row += f"  {'*' if vals[-1] == best else ''}"
        print(row)


# ============================================================
# 5. SAVE RESULTS
# ============================================================

def save_results(r_1x, r_2x):
    """Save comparison metrics to Drive as JSON."""
    if not r_1x and not r_2x:
        print("No results to save.")
        return

    comparison = {}
    if r_1x:
        comparison['c1_1x'] = r_1x
    if r_2x:
        comparison['c1_2x'] = r_2x

    comp_path = os.path.join(NOTEBOOK_DIR, 'outputs/comparison_metrics.json')
    os.makedirs(os.path.dirname(comp_path), exist_ok=True)

    with open(comp_path, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)

    print(f"\nComparison saved: {comp_path}")


# ============================================================
# 6. MAIN
# ============================================================

def main():
    print("DermaDiff - Evaluation Script")
    print("=" * 60)

    # Step 1: Run evaluations
    print("\nStep 1: Running evaluations on test set...")

    # C1-1x
    c1_1x_ckpt = os.path.join(C1_1X_OUTPUT, 'checkpoint-best.pth')
    if not os.path.exists(c1_1x_ckpt):
        c1_1x_ckpt = os.path.join(C1_1X_DRIVE, 'checkpoint-best.pth')

    if os.path.exists(c1_1x_ckpt):
        run_evaluation(c1_1x_ckpt, c1_1x_csv, C1_1X_IMAGES,
                      '/content/c1_1x_test_output', 'C1-1x')
    else:
        print("C1-1x checkpoint not found. Skip evaluation.")

    # C1-2x
    c1_2x_ckpt = os.path.join(C1_2X_OUTPUT, 'checkpoint-best.pth')
    if not os.path.exists(c1_2x_ckpt):
        c1_2x_ckpt = os.path.join(C1_2X_DRIVE, 'checkpoint-best.pth')

    if os.path.exists(c1_2x_ckpt):
        run_evaluation(c1_2x_ckpt, c1_2x_csv, C1_2X_IMAGES,
                      '/content/c1_2x_test_output', 'C1-2x')
    else:
        print("C1-2x checkpoint not found. Skip evaluation.")

    # Step 2: Compute metrics
    print("\nStep 2: Computing detailed metrics...")

    c1_1x_test_csv = '/content/c1_1x_test_output/test.csv'
    c1_2x_test_csv = '/content/c1_2x_test_output/test.csv'

    r_1x = compute_metrics(c1_1x_test_csv, 'C1-1x (SD 3.5 Large)')
    r_2x = compute_metrics(c1_2x_test_csv, 'C1-2x (SD 3.5 Large)')

    # Step 3: Cross-experiment comparison
    print("\nStep 3: Cross-experiment comparison...")
    compare_all_experiments(r_1x, r_2x)

    # Step 4: Save results
    print("\nStep 4: Saving results...")
    save_results(r_1x, r_2x)

    print("\nDone!")


if __name__ == '__main__':
    main()
