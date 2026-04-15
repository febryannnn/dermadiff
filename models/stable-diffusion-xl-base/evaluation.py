#!/usr/bin/env python3
"""
DermaDiff — Phase 4: Classifier Evaluation
============================================

Evaluates the PanDerm classifier from Phase 3 on the HAM10000 test set, then
computes detailed metrics: accuracy, weighted F1, macro F1, AUC-ROC, per-class
precision/recall/F1, and a confusion matrix.

This script wraps PanDerm's --eval mode (which writes a predictions CSV) and
then post-processes the output with scikit-learn.

Usage
-----
    python 4_evaluation.py \\
        --checkpoint /path/to/checkpoint-best.pth \\
        --csv_path /path/to/ham10000_c3_1x.csv \\
        --image_dir /path/to/combined_images \\
        --panderm_dir /path/to/PanDerm \\
        --output_dir /path/to/eval_output
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

# ────────────────────────────────────────────────────────────────────────
# config (must match phase 3 training)
# ────────────────────────────────────────────────────────────────────────

MODEL_NAME = "PanDerm_Large_FT"
NUM_CLASSES = 7
BATCH_SIZE = 128

LABEL_MAP = {
    "akiec": 0, "bcc": 1, "bkl": 2, "df": 3,
    "mel": 4, "nv": 5, "vasc": 6,
}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
CLASS_NAMES = [INV_LABEL_MAP[i] for i in range(NUM_CLASSES)]


def patch_panderm_torch_load(panderm_dir: str) -> None:
    """Patch PanDerm's run_class_finetuning.py for PyTorch 2.6+ compatibility.

    PyTorch 2.6 changed the default of `torch.load(weights_only=True)`, which
    refuses to load checkpoints containing numpy scalars — which PanDerm
    checkpoints do. We override by adding `weights_only=False` to the
    relevant calls. Idempotent.
    """
    finetune_file = os.path.join(panderm_dir, "classification/run_class_finetuning.py")
    if not os.path.exists(finetune_file):
        return

    with open(finetune_file, "r") as f:
        code = f.read()

    if "weights_only=False" in code:
        return  # already patched

    code = code.replace(
        "torch.load(model_weight)",
        "torch.load(model_weight, weights_only=False)",
    )
    code = code.replace(
        "torch.load(args.resume",
        "torch.load(args.resume, weights_only=False",
    )
    with open(finetune_file, "w") as f:
        f.write(code)
    print("Patched PanDerm torch.load calls for PyTorch 2.6+ compatibility")


def run_panderm_eval(
    panderm_dir: str,
    checkpoint: str,
    csv_path: str,
    image_dir: str,
    output_dir: str,
) -> int:
    """Invoke PanDerm's run_class_finetuning.py with --eval mode.

    Writes a `test.csv` to `output_dir` containing per-image predictions:
        filename, true_label, predicted_label, probability_class_0..6
    """
    os.makedirs(output_dir, exist_ok=True)
    os.environ["WANDB_MODE"] = "disabled"

    # apply the torch.load patch before calling PanDerm's script
    patch_panderm_torch_load(panderm_dir)

    classification_dir = os.path.join(panderm_dir, "classification")
    cmd = [
        "python3", "run_class_finetuning.py",
        "--model", MODEL_NAME,
        "--nb_classes", str(NUM_CLASSES),
        "--batch_size", str(BATCH_SIZE),
        "--sin_pos_emb",
        "--imagenet_default_mean_and_std",
        "--eval",
        "--resume", checkpoint,
        "--csv_path", csv_path,
        "--root_path", f"{image_dir}/",
        "--output_dir", output_dir,
    ]

    print(f"Running PanDerm evaluation...")
    print("  " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=classification_dir)
    return result.returncode


def compute_metrics(test_csv: str, label: str) -> dict:
    """Compute all classification metrics from PanDerm's eval CSV.

    Expected columns: filename, true_label, predicted_label,
                      probability_class_0..probability_class_6
    """
    df = pd.read_csv(test_csv)
    y_true = df["true_label"].values
    y_pred = df["predicted_label"].values

    prob_cols = [c for c in df.columns if c.startswith("probability_class_")]
    y_probs = df[prob_cols].values if prob_cols else None

    acc = accuracy_score(y_true, y_pred)
    w_f1 = f1_score(y_true, y_pred, average="weighted")
    m_f1 = f1_score(y_true, y_pred, average="macro")

    auc = None
    if y_probs is not None and y_probs.shape[1] == NUM_CLASSES:
        try:
            auc = roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
        except ValueError:
            pass

    prec_per = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    print(f"\n{'=' * 60}")
    print(f"  {label} — TEST SET RESULTS")
    print(f"{'=' * 60}")
    print(f"  Accuracy:     {acc:.4f}")
    print(f"  Weighted F1:  {w_f1:.4f}")
    print(f"  Macro F1:     {m_f1:.4f}")
    if auc is not None:
        print(f"  AUC-ROC:      {auc:.4f}")

    print(f"\n  PER-CLASS METRICS:")
    print(f"  {'Class':>6s}  {'Prec':>6s}  {'Recall':>6s}  {'F1':>6s}  {'N':>5s}")
    print(f"  {'-' * 38}")
    for i, cls in enumerate(CLASS_NAMES):
        n = int((y_true == i).sum())
        print(f"  {cls:>6s}  {prec_per[i]:>6.3f}  {rec_per[i]:>6.3f}  "
              f"{f1_per[i]:>6.3f}  {n:>5d}")

    print(f"\n  CONFUSION MATRIX (rows=true, cols=pred):")
    print(f"  {' ' * 6}", "  ".join(f"{c:>5s}" for c in CLASS_NAMES))
    for i, cls in enumerate(CLASS_NAMES):
        print(f"  {cls:>6s}", "  ".join(f"{cm[i, j]:>5d}" for j in range(NUM_CLASSES)))

    print(f"\n{classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3)}")

    return {
        "label": label,
        "accuracy": float(acc),
        "weighted_f1": float(w_f1),
        "macro_f1": float(m_f1),
        "auc_roc": float(auc) if auc is not None else None,
        "per_class": {
            cls: {
                "precision": float(prec_per[i]),
                "recall": float(rec_per[i]),
                "f1": float(f1_per[i]),
                "support": int((y_true == i).sum()),
            } for i, cls in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": cm.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DermaDiff PanDerm classifier on the test set"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Trained classifier checkpoint (e.g. checkpoint-best.pth)")
    parser.add_argument("--csv_path", required=True,
                        help="CSV used during training (must include test split rows)")
    parser.add_argument("--image_dir", required=True,
                        help="Directory containing all images referenced in the CSV")
    parser.add_argument("--panderm_dir", required=True,
                        help="Path to cloned PanDerm repository")
    parser.add_argument("--output_dir", required=True,
                        help="Directory where eval predictions and metrics are written")
    parser.add_argument("--label", default="DermaDiff",
                        help="Label printed in the report header")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip PanDerm eval; only post-process existing test.csv")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: checkpoint not found at {args.checkpoint}")
        sys.exit(1)

    if not args.skip_eval:
        rc = run_panderm_eval(
            args.panderm_dir, args.checkpoint, args.csv_path,
            args.image_dir, args.output_dir,
        )
        if rc != 0:
            print(f"ERROR: PanDerm eval failed (exit {rc})")
            sys.exit(rc)

    test_csv = os.path.join(args.output_dir, "test.csv")
    if not os.path.exists(test_csv):
        print(f"ERROR: PanDerm did not produce {test_csv}")
        print(f"Contents of {args.output_dir}: {os.listdir(args.output_dir)}")
        sys.exit(1)

    metrics = compute_metrics(test_csv, args.label)

    metrics_json = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved: {metrics_json}")


if __name__ == "__main__":
    main()
