#!/usr/bin/env python3
"""
DermaDiff — Cross-Domain Evaluation on PAD-UFES-20
====================================================

Evaluates trained PanDerm classifiers on the PAD-UFES-20 dataset to test
out-of-distribution (OOD) generalization. Each experiment's best checkpoint
is loaded and evaluated against PAD-UFES-20's 5 classes.

PAD-UFES-20 contains 2,106 clinical images across 5 skin lesion classes
(akiec, bcc, bkl, mel, nv). The classifiers were trained on HAM10000
(7 classes) — df and vasc have no PAD-UFES-20 samples.

Outputs
-------
- Per-experiment: accuracy, weighted F1, macro F1, per-class recall
- Confusion matrices (normalized, saved as PNG)
- CSV with all metrics
- Per-image predictions CSV

Usage
-----
    python evaluation/cross_domain_eval.py \\
        --pad_csv ./data/pad-ufes-20/pad_ufes_mapped.csv \\
        --pad_images ./data/pad-ufes-20/images \\
        --panderm_dir ./PanDerm \\
        --checkpoints \\
            "Exp A" ./outputs/classifier_expa/checkpoint-best.pth \\
            "Exp C" ./outputs/classifier_expc/checkpoint-best.pth \\
            "Exp E" ./outputs/classifier_expe/checkpoint-best.pth \\
        --output_dir ./outputs/cross_domain_eval
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(ALL_CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}
NUM_CLASSES = 7

# PAD-UFES-20 only has these 5 classes
PAD_CLASSES = ["akiec", "bcc", "bkl", "mel", "nv"]
PAD_LABEL_INDICES = [CLASS_TO_IDX[c] for c in PAD_CLASSES]

IMAGE_SIZE = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 64
NUM_WORKERS = 2


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PADUFESDataset(Dataset):
    """PAD-UFES-20 dataset from a CSV with img_id and dx columns."""

    def __init__(self, dataframe, img_lookup, transform):
        self.df = dataframe.reset_index(drop=True)
        self.img_lookup = img_lookup
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_lookup[row["img_id"]]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        label = CLASS_TO_IDX[row["dx"]]
        return img, label


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_panderm_checkpoint(ckpt_path, panderm_dir, device):
    """Load a PanDerm ViT-L classifier from a training checkpoint."""
    classification_dir = os.path.join(panderm_dir, "classification")
    if classification_dir not in sys.path:
        sys.path.insert(0, classification_dir)

    from models.modeling_finetune import panderm_large_patch16_224_finetune

    model = panderm_large_patch16_224_finetune(
        False,
        NUM_CLASSES,
        drop_rate=0.0,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_block_rate=None,
        use_mean_pooling=True,
        init_scale=0.001,
        use_rel_pos_bias=True,
        init_values=1e-5,
        lin_probe=False,
    )
    model.head = nn.Linear(model.head.in_features, NUM_CLASSES)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # Strip DataParallel 'module.' prefix if present
    cleaned = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_on_pad(model, loader, device):
    """Run inference and return predictions + labels."""
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="  Evaluating", leave=False):
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    return np.array(all_labels), np.array(all_preds)


def compute_metrics(y_true, y_pred):
    """Compute accuracy, F1, recall (overall + per-class)."""
    from sklearn.metrics import (
        accuracy_score, f1_score, recall_score, confusion_matrix,
    )

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

    per_class = {}
    for cls in PAD_CLASSES:
        idx = CLASS_TO_IDX[cls]
        mask = y_true == idx
        if mask.sum() == 0:
            continue
        cls_correct = int((y_pred[mask] == idx).sum())
        cls_total = int(mask.sum())
        per_class[cls] = {
            "recall": cls_correct / cls_total,
            "correct": cls_correct,
            "total": cls_total,
        }
    results["per_class"] = per_class

    cm = confusion_matrix(y_true, y_pred, labels=PAD_LABEL_INDICES)
    results["confusion_matrix"] = cm

    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_confusion_matrices(all_results, output_dir):
    """Save a grid of normalized confusion matrices, one per experiment."""
    n = len(all_results)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5.5 * rows))
    fig.patch.set_facecolor("white")

    if n == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    for i, (exp_name, res) in enumerate(all_results.items()):
        ax = axes[i]
        cm = res["confusion_matrix"]
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues",
                       vmin=0, vmax=1)
        ax.set_title(f"{exp_name}\nMacro F1 = {res['macro_f1']:.4f}",
                     fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(PAD_CLASSES)))
        ax.set_yticks(range(len(PAD_CLASSES)))
        ax.set_xticklabels(PAD_CLASSES, fontsize=9, rotation=45)
        ax.set_yticklabels(PAD_CLASSES, fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        for r in range(cm_norm.shape[0]):
            for c in range(cm_norm.shape[1]):
                color = "white" if cm_norm[r, c] > 0.5 else "black"
                ax.text(c, r, f"{cm_norm[r, c]:.2f}\n({cm[r, c]})",
                        ha="center", va="center", fontsize=8, color=color)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrices.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_per_class_recall(all_results, output_dir):
    """Grouped bar chart of per-class recall across experiments."""
    exp_names = list(all_results.keys())
    n_exp = len(exp_names)
    x = np.arange(len(PAD_CLASSES))
    bw = 0.8 / n_exp

    fig, ax = plt.subplots(figsize=(12, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, exp_name in enumerate(exp_names):
        recalls = []
        for cls in PAD_CLASSES:
            pc = all_results[exp_name].get("per_class", {}).get(cls)
            recalls.append(pc["recall"] if pc else 0.0)
        offset = (i - n_exp / 2 + 0.5) * bw
        bars = ax.bar(x + offset, recalls, width=bw * 0.88, label=exp_name,
                      edgecolor="white", linewidth=0.5, zorder=3)
        for bar, val in zip(bars, recalls):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=8, fontweight="medium")

    ax.set_ylabel("Recall", fontsize=13)
    ax.set_title("Per-Class Recall on PAD-UFES-20", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(PAD_CLASSES, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="-", alpha=0.3, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "per_class_recall.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def build_image_lookup(pad_img_base):
    """Build img_id -> full_path mapping across image subdirectories."""
    lookup = {}
    for root, _dirs, files in os.walk(pad_img_base):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in {".jpg", ".jpeg", ".png"}:
                img_id = os.path.splitext(fname)[0]
                lookup[img_id] = os.path.join(root, fname)
    return lookup


def save_predictions_csv(exp_name, df, y_true, y_pred, output_dir):
    """Save per-image predictions for one experiment."""
    pred_df = df[["img_id", "dx"]].copy()
    pred_df["label_idx"] = y_true
    pred_df["pred_idx"] = y_pred
    pred_df["pred_class"] = [IDX_TO_CLASS.get(p, "?") for p in y_pred]
    pred_df["correct"] = (y_true == y_pred).astype(int)

    safe_name = exp_name.replace(" ", "_").replace("/", "_").lower()
    path = os.path.join(output_dir, f"predictions_{safe_name}.csv")
    pred_df.to_csv(path, index=False)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate PanDerm classifiers on PAD-UFES-20.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pad_csv", required=True,
        help="CSV file with img_id and dx columns for PAD-UFES-20.",
    )
    parser.add_argument(
        "--pad_images", required=True,
        help="Root directory containing PAD-UFES-20 images (searched recursively).",
    )
    parser.add_argument(
        "--panderm_dir", required=True,
        help="Path to cloned PanDerm repository.",
    )
    parser.add_argument(
        "--checkpoints", required=True, nargs="+",
        help=(
            "Pairs of (name, path) for each checkpoint. "
            'E.g.: "Exp A" ./ckpt_a.pth "Exp C" ./ckpt_c.pth'
        ),
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory to write results (CSV, PNG, predictions).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE,
        help="Batch size for inference (default: 64).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: auto-detect).",
    )
    return parser.parse_args()


def _parse_pairs(raw):
    """Parse flat list ['name1', 'path1', 'name2', 'path2', ...] into pairs."""
    if len(raw) % 2 != 0:
        raise ValueError(
            "--checkpoints must be pairs of (name, path). Got odd number of arguments."
        )
    pairs = []
    for i in range(0, len(raw), 2):
        pairs.append((raw[i], raw[i + 1]))
    return pairs


def main():
    args = parse_args()
    checkpoints = _parse_pairs(args.checkpoints)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load PAD-UFES-20 ---
    print("Loading PAD-UFES-20 dataset ...")
    pad_df = pd.read_csv(args.pad_csv)
    img_lookup = build_image_lookup(args.pad_images)

    # Filter to images that exist and have valid class labels
    pad_df = pad_df[pad_df["dx"].isin(PAD_CLASSES)]
    pad_df = pad_df[pad_df["img_id"].isin(img_lookup)]
    pad_df = pad_df.reset_index(drop=True)
    print(f"  {len(pad_df)} images across {pad_df['dx'].nunique()} classes")
    print(f"  Class distribution: {dict(pad_df['dx'].value_counts())}")

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])

    dataset = PADUFESDataset(pad_df, img_lookup, eval_transform)
    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=NUM_WORKERS,
    )

    # --- Evaluate each checkpoint ---
    all_results = {}
    summary_rows = []

    for exp_name, ckpt_path in checkpoints:
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_name}")
        print(f"Checkpoint: {ckpt_path}")
        print(f"{'='*60}")

        if not os.path.isfile(ckpt_path):
            print(f"  SKIP — checkpoint not found")
            continue

        model = load_panderm_checkpoint(ckpt_path, args.panderm_dir, device)
        y_true, y_pred = evaluate_on_pad(model, loader, device)
        del model
        torch.cuda.empty_cache()

        res = compute_metrics(y_true, y_pred)
        all_results[exp_name] = res

        # Print summary
        print(f"  Accuracy:      {res['accuracy']:.4f}")
        print(f"  Weighted F1:   {res['weighted_f1']:.4f}")
        print(f"  Macro F1:      {res['macro_f1']:.4f}")
        print(f"  Per-class recall:")
        for cls in PAD_CLASSES:
            pc = res["per_class"].get(cls)
            if pc:
                print(f"    {cls:>6}: {pc['recall']:.4f} ({pc['correct']}/{pc['total']})")

        # CSV row
        row = {
            "experiment": exp_name,
            "accuracy": res["accuracy"],
            "weighted_f1": res["weighted_f1"],
            "macro_f1": res["macro_f1"],
            "weighted_recall": res["weighted_recall"],
            "macro_recall": res["macro_recall"],
        }
        for cls in PAD_CLASSES:
            pc = res["per_class"].get(cls)
            row[f"recall_{cls}"] = pc["recall"] if pc else ""
        summary_rows.append(row)

        # Per-image predictions
        save_predictions_csv(exp_name, pad_df, y_true, y_pred, args.output_dir)

    if not all_results:
        print("\nNo checkpoints were evaluated.")
        return

    # --- Save summary CSV ---
    csv_path = os.path.join(args.output_dir, "cross_domain_results.csv")
    fieldnames = (
        ["experiment", "accuracy", "weighted_f1", "macro_f1",
         "weighted_recall", "macro_recall"]
        + [f"recall_{c}" for c in PAD_CLASSES]
    )
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSaved: {csv_path}")

    # --- Plots ---
    plot_confusion_matrices(all_results, args.output_dir)
    plot_per_class_recall(all_results, args.output_dir)

    # --- Final summary table ---
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}\n")
    header = f"{'Experiment':<20} {'Acc':>6} {'W-F1':>6} {'M-F1':>6}"
    for cls in PAD_CLASSES:
        header += f" {cls:>6}"
    print(header)
    print("-" * len(header))
    for exp_name, res in all_results.items():
        line = f"{exp_name:<20} {res['accuracy']:>6.4f} {res['weighted_f1']:>6.4f} {res['macro_f1']:>6.4f}"
        for cls in PAD_CLASSES:
            pc = res["per_class"].get(cls)
            line += f" {pc['recall']:>6.4f}" if pc else f" {'N/A':>6}"
        print(line)


if __name__ == "__main__":
    main()
