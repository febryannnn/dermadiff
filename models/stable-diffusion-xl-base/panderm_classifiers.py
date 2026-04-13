#!/usr/bin/env python3
"""
DermaDiff — Phase 3: PanDerm Classifier Training
==================================================

Trains the PanDerm ViT-Large classifier on HAM10000 augmented with synthetic
images from Phase 2. Builds the training CSV in PanDerm's expected format,
assembles a combined image directory (real + synthetic), and invokes PanDerm's
`run_class_finetuning.py` script with hyperparameters identical to the baseline.

Usage
-----
    python 3_classifier_training.py \\
        --ham_images /path/to/ham10000/images \\
        --ham_metadata /path/to/HAM10000_metadata.csv \\
        --splits_json /path/to/ham10000_splits.json \\
        --synthetic_dir /path/to/synthetic_images \\
        --panderm_dir /path/to/PanDerm \\
        --panderm_weights /path/to/panderm_pretrained.pth \\
        --output_dir /path/to/classifier_output \\
        --ratio 1
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys

import pandas as pd

# ────────────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────────────

MODEL_NAME = "PanDerm_Large_FT"
NUM_CLASSES = 7
BATCH_SIZE = 128
LR = 5e-4
WARMUP_EPOCHS = 10
EPOCHS = 50
LAYER_DECAY = 0.65
DROP_PATH = 0.2
WEIGHT_DECAY = 0.05
MIXUP = 0.8
CUTMIX = 1.0
SEED = 0
SPLIT_SEED = 42

# Label mapping must match what PanDerm was pretrained with
LABEL_MAP = {
    "akiec": 0, "bcc": 1, "bkl": 2, "df": 3,
    "mel": 4, "nv": 5, "vasc": 6,
}

TARGET_CLASSES = ["mel", "bcc", "akiec", "df", "vasc"]


def patch_panderm_torch_load(panderm_dir: str) -> None:
    """Patch PanDerm's run_class_finetuning.py for PyTorch 2.6+ compatibility.

    Newer PyTorch defaults `torch.load(weights_only=True)` which breaks loading
    of full PanDerm checkpoints (they contain optimizer state). We override
    that by adding `weights_only=False` to the relevant calls.
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
        "torch.load(model_weight, weights_only=False)"
    )
    code = code.replace(
        "torch.load(args.resume",
        "torch.load(args.resume, weights_only=False"
    )
    with open(finetune_file, "w") as f:
        f.write(code)
    print("Patched PanDerm torch.load calls for PyTorch 2.6+ compatibility")


def build_csv(
    ham_metadata: str,
    splits_json: str,
    synthetic_dir: str,
    output_csv: str,
    ratio: int,
) -> pd.DataFrame:
    """Build PanDerm-format CSV with columns: image, label, split.

    Real HAM10000 images use their original train/val/test split. Synthetic
    images are appended with split='train' only — never val or test.
    """
    with open(splits_json) as f:
        splits = json.load(f)
    train_ids = set(splits["train"])
    val_ids = set(splits["val"])
    test_ids = set(splits["test"])
    id_col = splits["metadata"]["id_column"]

    ham_df = pd.read_csv(ham_metadata)

    records = []
    for _, row in ham_df.iterrows():
        img_id = row[id_col]
        dx = row["dx"]
        if dx not in LABEL_MAP:
            continue

        if img_id in train_ids:
            split = "train"
        elif img_id in val_ids:
            split = "val"
        elif img_id in test_ids:
            split = "test"
        else:
            continue

        records.append({
            "image": f"{img_id}.jpg",
            "label": LABEL_MAP[dx],
            "split": split,
        })

    base_df = pd.DataFrame(records)
    print(f"  Real images: train={(base_df['split']=='train').sum()}, "
          f"val={(base_df['split']=='val').sum()}, "
          f"test={(base_df['split']=='test').sum()}")

    # Append synthetic images with subsampling for the requested ratio
    random.seed(SPLIT_SEED)
    synth_records = []
    for cls in TARGET_CLASSES:
        cls_dir = os.path.join(synthetic_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"  WARNING: no synthetic directory for {cls}")
            continue

        all_synth = sorted([
            f for f in os.listdir(cls_dir) if f.endswith(".jpg")
        ])
        train_count = int((ham_df[ham_df[id_col].isin(train_ids)]["dx"] == cls).sum())
        target = train_count * ratio
        subset = random.sample(all_synth, min(target, len(all_synth)))

        for fname in subset:
            synth_records.append({
                "image": fname,
                "label": LABEL_MAP[cls],
                "split": "train",
            })
        print(f"  Synthetic {cls}: kept {len(subset)} (target {target}, available {len(all_synth)})")

    full_df = pd.concat([base_df, pd.DataFrame(synth_records)], ignore_index=True)
    full_df.to_csv(output_csv, index=False)
    print(f"  Wrote CSV: {output_csv} ({len(full_df)} rows)")
    return full_df


def build_combined_image_dir(
    ham_images: str,
    synthetic_dir: str,
    df: pd.DataFrame,
    combined_dir: str,
) -> None:
    """Symlink HAM10000 images and copy synthetic images into one root_path.

    PanDerm's training script reads all images from a single --root_path, so
    real and synthetic images must live in the same directory.
    """
    os.makedirs(combined_dir, exist_ok=True)
    existing = set(os.listdir(combined_dir))

    # Symlink real HAM images (fast — no copy)
    ham_count = 0
    for fname in os.listdir(ham_images):
        if fname in existing or not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        os.symlink(
            os.path.join(ham_images, fname),
            os.path.join(combined_dir, fname),
        )
        ham_count += 1

    # Copy synthetic images that are referenced in the CSV
    synth_filenames = set(df["image"].tolist()) - existing
    synth_count = 0
    for cls in TARGET_CLASSES:
        cls_dir = os.path.join(synthetic_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname in synth_filenames:
                shutil.copy2(
                    os.path.join(cls_dir, fname),
                    os.path.join(combined_dir, fname),
                )
                synth_count += 1

    total = len(os.listdir(combined_dir))
    print(f"  Combined dir: HAM symlinks={ham_count}, synth copies={synth_count}, total={total}")


def train_classifier(
    panderm_dir: str,
    panderm_weights: str,
    csv_path: str,
    image_dir: str,
    output_dir: str,
    exp_name: str,
) -> int:
    """Invoke PanDerm's run_class_finetuning.py with shared hyperparameters."""
    os.makedirs(output_dir, exist_ok=True)
    os.environ["WANDB_MODE"] = "disabled"

    classification_dir = os.path.join(panderm_dir, "classification")
    cmd = [
        "python3", "run_class_finetuning.py",
        "--model", MODEL_NAME,
        "--pretrained_checkpoint", panderm_weights,
        "--nb_classes", str(NUM_CLASSES),
        "--batch_size", str(BATCH_SIZE),
        "--lr", str(LR),
        "--update_freq", "1",
        "--warmup_epochs", str(WARMUP_EPOCHS),
        "--epochs", str(EPOCHS),
        "--layer_decay", str(LAYER_DECAY),
        "--drop_path", str(DROP_PATH),
        "--weight_decay", str(WEIGHT_DECAY),
        "--mixup", str(MIXUP),
        "--cutmix", str(CUTMIX),
        "--weights",
        "--sin_pos_emb",
        "--no_auto_resume",
        "--imagenet_default_mean_and_std",
        "--exp_name", exp_name,
        "--output_dir", output_dir,
        "--csv_path", csv_path,
        "--root_path", f"{image_dir}/",
        "--seed", str(SEED),
    ]

    print(f"  Running: cd {classification_dir}")
    print("  " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=classification_dir)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Train PanDerm classifier on HAM10000 + SDXL LoRA synthetics"
    )
    parser.add_argument("--ham_images", required=True,
                        help="Directory containing HAM10000 image files")
    parser.add_argument("--ham_metadata", required=True,
                        help="HAM10000_metadata.csv path")
    parser.add_argument("--splits_json", required=True,
                        help="JSON with train/val/test image ID lists")
    parser.add_argument("--synthetic_dir", required=True,
                        help="Directory of generated synthetic images (per-class subdirs)")
    parser.add_argument("--panderm_dir", required=True,
                        help="Path to cloned PanDerm repository")
    parser.add_argument("--panderm_weights", required=True,
                        help="Pretrained PanDerm checkpoint (.pth)")
    parser.add_argument("--output_dir", required=True,
                        help="Directory for trained classifier checkpoint")
    parser.add_argument("--ratio", type=int, default=1,
                        help="Synthetic-to-real ratio (default: 1)")
    parser.add_argument("--temp_dir", default="/tmp/dermadiff_classifier",
                        help="Working directory for combined images and CSV")
    parser.add_argument("--exp_name", default="DermaDiff_C3",
                        help="Experiment name passed to PanDerm")
    args = parser.parse_args()

    os.makedirs(args.temp_dir, exist_ok=True)
    patch_panderm_torch_load(args.panderm_dir)

    print(f"\n{'=' * 60}")
    print(f"  Building training CSV (ratio={args.ratio}x)")
    print(f"{'=' * 60}")
    csv_path = os.path.join(args.temp_dir, f"ham10000_c3_{args.ratio}x.csv")
    df = build_csv(
        args.ham_metadata, args.splits_json, args.synthetic_dir,
        csv_path, args.ratio
    )

    print(f"\n{'=' * 60}")
    print(f"  Building combined image directory")
    print(f"{'=' * 60}")
    combined_dir = os.path.join(args.temp_dir, f"images_c3_{args.ratio}x")
    build_combined_image_dir(args.ham_images, args.synthetic_dir, df, combined_dir)

    print(f"\n{'=' * 60}")
    print(f"  Training classifier")
    print(f"{'=' * 60}")
    rc = train_classifier(
        args.panderm_dir, args.panderm_weights,
        csv_path, combined_dir, args.output_dir,
        f"{args.exp_name}_{args.ratio}x",
    )

    if rc == 0:
        print(f"\nClassifier training complete: {args.output_dir}")
    else:
        print(f"\nClassifier training failed with exit code {rc}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
