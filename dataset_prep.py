#!/usr/bin/env python3
"""
DermaDiff - Phase 0: Dataset Preparation
==========================================

Prepares the per-class training image pool used by Phase 1 (LoRA fine-tuning).

This script does four things:

1. Builds HAM10000 stratified splits: 70/15/15 train/val/test by 'dx' label,
   saved as JSON with image IDs (used by Phase 3 classifier training).

2. **Extracts longitudinal minority images: walks the longitudinal Excel
   metadata, maps each image to its 'Diagnosis' column, and organizes them
   into per-class folders matching HAM10000 label names (mel, bcc, akiec, df).

3. Builds the combined per-class training pool: for Phase 1, applying the
   filtering rule: HAM10000 train split only (no val/test leakage), but ALL
   ISIC 2019 and longitudinal images.

4. Symlinks instead of copies: no disk space duplication. The output
   directory contains symlinks pointing to the original files in their
   source locations.

Usage
-----
    python dataset_prep.py \\
        --ham_images /path/to/ham10000/images \\
        --ham_metadata /path/to/HAM10000_metadata.csv \\
        --isic_images /path/to/isic2019/images \\
        --longitudinal_dir /path/to/longitudinal \\
        --longitudinal_metadata "/path/to/HighRisk Dermoscopic images.xlsx" \\
                                "/path/to/General Dermosopic images.xlsx" \\
        --output_splits /path/to/ham10000_splits.json \\
        --output_per_class_dir /path/to/training_images_per_class \\
        --output_longitudinal_dir /path/to/longitudinal_extracted

The `--output_per_class_dir` is what you pass as `--train_data_dir` to Phase 1.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --------
# CONFIG 
# --------

SPLIT_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# the 5 minority classes targeted for LoRA training and synthetic generation
TARGET_CLASSES = ["mel", "bcc", "akiec", "df", "vasc"]

# ISIC 2019 folder name to HAM10000 label
ISIC_TO_HAM = {
    "MEL (malignant)":  "mel",
    "BCC (malignant)":  "bcc",
    "AK (suspicious)":  "akiec",
    "BKL (low risk)":   "bkl",
    "DF (low risk)":    "df",
    "NV (low risk)":    "nv",
    "VASC (low risk)":  "vasc",
}

# longitudinal Diagnosis column to HAM10000 label
LONGITUDINAL_TO_HAM = {
    "melanoma":             "mel",
    "basal cell carcinoma": "bcc",
    "actinic keratosis":    "akiec",
    "dermatofibroma":       "df",
}

# excel column containing the longitudinal image filenames
LONGITUDINAL_IMG_COL = "Dermoscopic_Image_ID*(ParticipantID_LesionID_visitID)"

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# -----------------------------
#  HAM10000 STRATIFIED SPLIT
# -----------------------------

def build_ham_splits(ham_metadata: str, output_splits: str) -> dict:
    """Build a 70/15/15 stratified train/val/test split for HAM10000.

    Saves splits as JSON with image_id lists. Skips if file already exists.
    """
    if os.path.exists(output_splits):
        print(f"  Split file already exists: {output_splits}")
        with open(output_splits) as f:
            splits = json.load(f)
        print(f"  Loaded existing: train={len(splits['train'])}, "
              f"val={len(splits['val'])}, test={len(splits['test'])}")
        return splits

    ham_df = pd.read_csv(ham_metadata)
    id_col = "image_id" if "image_id" in ham_df.columns else ham_df.columns[0]
    labels = ham_df["dx"].values
    indices = np.arange(len(ham_df))

    # first split: train vs (val + test)
    train_idx, temp_idx = train_test_split(
        indices, test_size=(VAL_RATIO + TEST_RATIO),
        stratify=labels, random_state=SPLIT_SEED,
    )
    # second split: val vs test
    relative_test = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=relative_test,
        stratify=labels[temp_idx], random_state=SPLIT_SEED,
    )

    splits = {
        "train": ham_df.iloc[train_idx][id_col].tolist(),
        "val":   ham_df.iloc[val_idx][id_col].tolist(),
        "test":  ham_df.iloc[test_idx][id_col].tolist(),
        "metadata": {
            "seed": SPLIT_SEED,
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "total_images": len(ham_df),
            "id_column": id_col,
            "created_at": pd.Timestamp.now().isoformat(),
        },
    }

    os.makedirs(os.path.dirname(output_splits), exist_ok=True)
    with open(output_splits, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"  Train: {len(splits['train'])} ({TRAIN_RATIO*100:.0f}%)")
    print(f"  Val:   {len(splits['val'])} ({VAL_RATIO*100:.0f}%)")
    print(f"  Test:  {len(splits['test'])} ({TEST_RATIO*100:.0f}%)")
    print(f"  Saved: {output_splits}")
    return splits


# -------------------------
# LONGITUDINAL EXTRACTION
# -------------------------

def extract_longitudinal(
    longitudinal_dir: str,
    metadata_files: list,
    output_dir: str,
) -> dict:
    """Extract longitudinal images, organized into per-class subdirectories.

    Walks the Excel metadata files, maps Diagnosis to HAM10000 label, finds
    the actual image file by walking longitudinal_dir, then symlinks each
    image into output_dir/{class}/.
    """
    if not metadata_files:
        print("  No longitudinal metadata files provided — skipping")
        return {}

    # load and combine metadata sheets
    dfs = []
    for meta_file in metadata_files:
        if not os.path.exists(meta_file):
            print(f"  WARNING: metadata file not found: {meta_file}")
            continue
        dfs.append(pd.read_excel(meta_file))
    if not dfs:
        return {}
    all_meta = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(all_meta)} longitudinal metadata entries")

    # build filename to full path lookup
    print(f"  Scanning {longitudinal_dir} for image files...")
    img_lookup = {}
    for root, _, files in os.walk(longitudinal_dir):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                img_lookup[f] = os.path.join(root, f)
    print(f"  Found {len(img_lookup)} candidate images on disk")

    # map each metadata row to a real image path
    if LONGITUDINAL_IMG_COL not in all_meta.columns:
        print(f"  ERROR: column {LONGITUDINAL_IMG_COL!r} not found in metadata")
        print(f"  Available columns: {list(all_meta.columns)}")
        return {}

    all_meta["image_path"] = all_meta[LONGITUDINAL_IMG_COL].map(img_lookup)
    matched = all_meta["image_path"].notna().sum()
    print(f"  Matched {matched}/{len(all_meta)} images to file paths")

    # filter to relevant minority classes
    relevant = all_meta[
        all_meta["Diagnosis"].isin(LONGITUDINAL_TO_HAM.keys())
        & all_meta["image_path"].notna()
    ].copy()
    relevant["dx"] = relevant["Diagnosis"].map(LONGITUDINAL_TO_HAM)
    print(f"  Relevant minority entries: {len(relevant)}")

    # symlink each image into per-class folders
    counts = {cls: 0 for cls in set(LONGITUDINAL_TO_HAM.values())}
    for _, row in relevant.iterrows():
        cls = row["dx"]
        cls_dir = os.path.join(output_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        src = row["image_path"]
        dst = os.path.join(cls_dir, os.path.basename(src))
        if os.path.exists(dst):
            counts[cls] += 1
            continue
        try:
            os.symlink(src, dst)
            counts[cls] += 1
        except OSError as e:
            print(f"  WARNING: failed to symlink {src}: {e}")

    print(f"  Extracted per class: {counts}")
    return counts


# --------------------------------------------
# BUILD COMBINED PER-CLASS TRAINING POOL
# --------------------------------------------

def build_training_pool(
    ham_images: str,
    ham_metadata: str,
    splits: dict,
    isic_images: str,
    longitudinal_dir: str,
    output_dir: str,
) -> dict:
    """Build the per-class training pool used by Phase 1 LoRA fine-tuning.

    Filtering rules:
      - HAM10000: train split ONLY (prevent val/test leakage into the diffusion model)
      - ISIC 2019: ALL images
      - Longitudinal: ALL extracted images

    Output layout:
      output_dir/
        mel/   <symlinks to all mel images from HAM-train + ISIC + longitudinal>
        bcc/
        akiec/
        df/
        vasc/
    """
    ham_df = pd.read_csv(ham_metadata)
    id_col = splits["metadata"]["id_column"]
    train_ids = set(splits["train"])
    ham_train_df = ham_df[ham_df[id_col].isin(train_ids)]

    counts = {cls: 0 for cls in TARGET_CLASSES}

    # Per-class output directories
    for cls in TARGET_CLASSES:
        os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

    # HAM10000
    print("  Symlinking HAM10000 train-split images...")
    for _, row in ham_train_df.iterrows():
        dx = row["dx"]
        if dx not in TARGET_CLASSES:
            continue
        img_id = row[id_col]
        src = os.path.join(ham_images, f"{img_id}.jpg")
        if not os.path.exists(src):
            continue
        dst = os.path.join(output_dir, dx, f"{img_id}.jpg")
        if os.path.exists(dst):
            counts[dx] += 1
            continue
        os.symlink(src, dst)
        counts[dx] += 1
    print(f"    HAM10000 contribution: { {k: v for k, v in counts.items()} }")

    # ISIC 2019
    if isic_images and os.path.isdir(isic_images):
        print("  Symlinking ISIC 2019 images...")
        before = dict(counts)
        for isic_folder, ham_label in ISIC_TO_HAM.items():
            if ham_label not in TARGET_CLASSES:
                continue
            cls_dir = os.path.join(isic_images, isic_folder)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if not fname.lower().endswith(IMG_EXTS):
                    continue
                src = os.path.join(cls_dir, fname)
                dst = os.path.join(output_dir, ham_label, f"isic_{fname}")
                if os.path.exists(dst):
                    counts[ham_label] += 1
                    continue
                os.symlink(src, dst)
                counts[ham_label] += 1
        added = {k: counts[k] - before[k] for k in TARGET_CLASSES}
        print(f"    ISIC contribution: {added}")
    else:
        print("  ISIC 2019 directory not found — skipping")

    # Longitudinal
    if longitudinal_dir and os.path.isdir(longitudinal_dir):
        print("  Symlinking longitudinal extracted images...")
        before = dict(counts)
        for cls in TARGET_CLASSES:
            cls_dir = os.path.join(longitudinal_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if not fname.lower().endswith(IMG_EXTS):
                    continue
                src = os.path.join(cls_dir, fname)
                if os.path.islink(src):
                    src = os.path.realpath(src)
                dst = os.path.join(output_dir, cls, f"long_{fname}")
                if os.path.exists(dst):
                    counts[cls] += 1
                    continue
                os.symlink(src, dst)
                counts[cls] += 1
        added = {k: counts[k] - before[k] for k in TARGET_CLASSES}
        print(f"    Longitudinal contribution: {added}")
    else:
        print("  Longitudinal directory not found — skipping")

    return counts


# --------------
# MAIN
# --------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare per-class training pool for DermaDiff Phase 1 LoRA fine-tuning"
    )
    parser.add_argument("--ham_images", required=True,
                        help="Directory containing HAM10000 .jpg files")
    parser.add_argument("--ham_metadata", required=True,
                        help="HAM10000_metadata.csv path")
    parser.add_argument("--isic_images", default=None,
                        help="ISIC 2019 root directory with class subfolders (optional)")
    parser.add_argument("--longitudinal_dir", default=None,
                        help="Longitudinal dataset root (will be walked for image files)")
    parser.add_argument("--longitudinal_metadata", nargs="+", default=[],
                        help="Longitudinal Excel metadata files (.xlsx)")
    parser.add_argument("--output_splits", required=True,
                        help="Output JSON path for HAM10000 train/val/test splits")
    parser.add_argument("--output_per_class_dir", required=True,
                        help="Output directory for per-class training pool (Phase 1 input)")
    parser.add_argument("--output_longitudinal_dir", default=None,
                        help="Output directory for extracted longitudinal images (per-class)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Phase 0: Dataset Preparation")
    print("=" * 60)

    # step 1: HAM10000 splits
    print("\n[1/3] Building HAM10000 stratified splits...")
    splits = build_ham_splits(args.ham_metadata, args.output_splits)

    # step 2: longitudinal extraction 
    if args.longitudinal_dir and args.longitudinal_metadata:
        print("\n[2/3] Extracting longitudinal minority images...")
        long_dir = args.output_longitudinal_dir or os.path.join(
            os.path.dirname(args.output_per_class_dir), "longitudinal_extracted"
        )
        extract_longitudinal(
            args.longitudinal_dir, args.longitudinal_metadata, long_dir
        )
    else:
        print("\n[2/3] Skipping longitudinal extraction (no metadata provided)")
        long_dir = None

    # step 3: build the combined per-class training pool
    print("\n[3/3] Building per-class training pool for Phase 1...")
    counts = build_training_pool(
        ham_images=args.ham_images,
        ham_metadata=args.ham_metadata,
        splits=splits,
        isic_images=args.isic_images,
        longitudinal_dir=long_dir,
        output_dir=args.output_per_class_dir,
    )

    print(f"\n{'=' * 60}")
    print(f"  DONE — Per-class training pool: {args.output_per_class_dir}")
    print(f"{'=' * 60}")
    total = 0
    for cls in TARGET_CLASSES:
        print(f"  {cls:6s}: {counts[cls]:5d} images")
        total += counts[cls]
    print(f"  Total: {total} images")
    print()
    print(f"Next: pass this directory as --train_data_dir to 1_finetune_lora.py")


if __name__ == "__main__":
    main()
