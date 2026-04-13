#!/usr/bin/env python3
"""
DermaDiff — Phase 1: SDXL LoRA Fine-tuning
============================================

Fine-tunes Stable Diffusion XL with LoRA adapters on dermoscopic skin lesion
images, producing one LoRA per minority class (mel, bcc, akiec, df, vasc).

This script wraps HuggingFace diffusers'
`examples/text_to_image/train_text_to_image_lora_sdxl.py` — the same training
script used to produce the published DermaDiff C3 results (Macro F1 = 0.8409).

The text-to-image script reads captions from a per-class `metadata.jsonl` file
where every image in a class shares the same dermoscopic caption. This script
auto-generates that metadata.jsonl for each class before invoking training,
then trains all five classes sequentially with adaptive step counts.

Usage
-----
    python fine_tuned_LoRA.py \\
        --train_data_dir /path/to/training_images \\
        --output_dir "/path/to/LoRA Weights" \\
        --diffusers_dir /path/to/diffusers_repo

Each class subdirectory under --train_data_dir must contain dermoscopic images
for that class (e.g. train_data_dir/mel/*.jpg, train_data_dir/bcc/*.jpg, ...).
The metadata.jsonl files are created automatically inside each class folder.
"""

import argparse
import json
import os
import subprocess
import sys
import time

# ────────────────────────────────────────────────────────────────────────
# CONFIG (hardcoded — change here, not via CLI)
# ────────────────────────────────────────────────────────────────────────

SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_FIX = "madebyollin/sdxl-vae-fp16-fix"

LORA_RANK = 16
LEARNING_RATE = 1e-4
RESOLUTION = 1024          # Native SDXL resolution (best results)
TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4
LR_SCHEDULER = "constant"
LR_WARMUP_STEPS = 100
SEED = 42

TARGET_CLASSES = ["mel", "bcc", "akiec", "df", "vasc"]

# One caption per class — written into metadata.jsonl alongside each image.
# Every image in a class receives this same caption (matches the published
# C3 LoRA training setup).
CLASS_CAPTIONS = {
    "mel":   "a dermoscopic photograph of a melanoma skin lesion",
    "bcc":   "a dermoscopic photograph of a basal cell carcinoma skin lesion",
    "akiec": "a dermoscopic photograph of an actinic keratosis skin lesion",
    "df":    "a dermoscopic photograph of a dermatofibroma skin lesion",
    "vasc":  "a dermoscopic photograph of a vascular skin lesion",
}


def get_max_steps(num_images: int) -> int:
    """Adaptive step count based on dataset size.

    Small classes (df ~94, vasc ~99) get more epochs to learn the concept,
    while large classes (mel ~5000+) train for fewer steps to avoid overfitting.
    """
    if num_images < 150:
        return 1500   # ~6 epochs for tiny classes
    elif num_images < 400:
        return 1000   # ~3 epochs for medium classes
    else:
        return 800    # ~2 epochs for large classes


def build_metadata_jsonl(cls_data_dir: str, caption: str) -> int:
    """Create metadata.jsonl inside a class directory.

    The diffusers text-to-image LoRA training script reads this file to
    associate each image with its caption. Format: one JSON object per line:
        {"file_name": "ISIC_0024306.jpg", "text": "a dermoscopic photograph..."}

    Returns the number of metadata entries written.
    """
    metadata_path = os.path.join(cls_data_dir, "metadata.jsonl")
    count = 0
    with open(metadata_path, "w") as f:
        for img_file in sorted(os.listdir(cls_data_dir)):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                entry = {"file_name": img_file, "text": caption}
                f.write(json.dumps(entry) + "\n")
                count += 1
    return count


def resolve_train_script(diffusers_dir):
    """Locate the LoRA training script.

    Priority order:
    1. If --diffusers_dir was provided → use that clone's examples/ folder
    2. Otherwise → use the bundled script next to this wrapper
       (models/stable-diffusion-xl-base/train_text_to_image_lora_sdxl.py)
    3. Return None if neither is found.

    The bundled script is sourced from diffusers v0.37.1 and is pinned as
    part of the repo for full reproducibility. Pass --diffusers_dir only if
    you want to override with a newer/different version.
    """
    if diffusers_dir:
        candidate = os.path.join(
            diffusers_dir, "examples/text_to_image/train_text_to_image_lora_sdxl.py"
        )
        if os.path.exists(candidate):
            return candidate
        print(f"  WARNING: --diffusers_dir was given but no script at {candidate}")
        return None

    # Bundled script lives next to this wrapper file
    bundled = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "train_text_to_image_lora_sdxl.py",
    )
    if os.path.exists(bundled):
        return bundled

    return None


def train_lora_for_class(
    cls_name: str,
    train_data_dir: str,
    output_dir: str,
    train_script: str,
) -> bool:
    """Train one SDXL LoRA adapter for a single skin lesion class."""

    cls_data_dir = os.path.join(train_data_dir, cls_name)
    cls_output_dir = os.path.join(output_dir, f"lora_{cls_name}_final")
    os.makedirs(cls_output_dir, exist_ok=True)

    # Skip if already trained
    weights_file = os.path.join(cls_output_dir, "pytorch_lora_weights.safetensors")
    if os.path.exists(weights_file):
        size_mb = os.path.getsize(weights_file) / 1024 / 1024
        print(f"  SKIP {cls_name}: already trained ({size_mb:.1f} MB)")
        return True

    if not os.path.isdir(cls_data_dir):
        print(f"  SKIP {cls_name}: directory not found at {cls_data_dir}")
        return False

    # Auto-generate metadata.jsonl for this class (required by text_to_image script)
    caption = CLASS_CAPTIONS[cls_name]
    n_images = build_metadata_jsonl(cls_data_dir, caption)
    if n_images == 0:
        print(f"  SKIP {cls_name}: no images in {cls_data_dir}")
        return False

    max_steps = get_max_steps(n_images)

    print(f"\n{'=' * 60}")
    print(f"  Training LoRA for: {cls_name}")
    print(f"  Images: {n_images} | Steps: {max_steps}")
    print(f"  Caption: {caption}")
    print(f"{'=' * 60}")

    cmd = [
        "accelerate", "launch", train_script,
        f"--pretrained_model_name_or_path={SDXL_MODEL}",
        f"--pretrained_vae_model_name_or_path={VAE_FIX}",
        f"--train_data_dir={cls_data_dir}",
        f"--caption_column=text",
        f"--output_dir={cls_output_dir}",
        f"--resolution={RESOLUTION}",
        f"--center_crop",
        f"--random_flip",
        f"--train_batch_size={TRAIN_BATCH_SIZE}",
        f"--gradient_accumulation_steps={GRADIENT_ACCUMULATION}",
        f"--learning_rate={LEARNING_RATE}",
        f"--lr_scheduler={LR_SCHEDULER}",
        f"--lr_warmup_steps={LR_WARMUP_STEPS}",
        f"--max_train_steps={max_steps}",
        f"--rank={LORA_RANK}",
        f"--seed={SEED}",
        "--mixed_precision=fp16",
        "--checkpointing_steps=500",
        "--report_to=tensorboard",
        "--dataloader_num_workers=4",
    ]

    start = time.time()
    result = subprocess.run(cmd, text=True)
    elapsed_min = (time.time() - start) / 60

    if result.returncode == 0 and os.path.exists(weights_file):
        size_mb = os.path.getsize(weights_file) / 1024 / 1024
        print(f"  SUCCESS: {cls_name} LoRA saved ({size_mb:.1f} MB) in {elapsed_min:.1f} min")
        return True

    print(f"  FAILED: {cls_name} (exit {result.returncode}) after {elapsed_min:.1f} min")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune SDXL with LoRA on dermoscopic skin lesion images "
                    "(uses diffusers text_to_image_lora_sdxl.py training script)"
    )
    parser.add_argument(
        "--train_data_dir", required=True,
        help="Root directory containing per-class subdirectories of training images"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Output directory for LoRA weights (one lora_{class}_final/ subdir per class)"
    )
    parser.add_argument(
        "--diffusers_dir", default=None,
        help="Optional path to a cloned huggingface/diffusers repository. "
             "If not provided, the script uses the bundled "
             "train_text_to_image_lora_sdxl.py (sourced from diffusers v0.37.1) "
             "that lives next to this wrapper."
    )
    parser.add_argument(
        "--classes", nargs="+", default=TARGET_CLASSES,
        help=f"Which classes to train (default: {' '.join(TARGET_CLASSES)})"
    )
    args = parser.parse_args()

    # Resolve the training script once, up front — fail fast if missing
    train_script = resolve_train_script(args.diffusers_dir)
    if train_script is None:
        print("ERROR: could not find a LoRA training script.")
        print("  Either:")
        print("  1. Place train_text_to_image_lora_sdxl.py next to this wrapper, OR")
        print("  2. Pass --diffusers_dir pointing at a cloned diffusers repo")
        sys.exit(1)
    print(f"Using training script: {train_script}")

    os.makedirs(args.output_dir, exist_ok=True)

    print("SDXL LoRA TRAINING PLAN")
    print("=" * 60)
    print(f"Base model:      {SDXL_MODEL}")
    print(f"Training script: train_text_to_image_lora_sdxl.py")
    print(f"LoRA rank:       {LORA_RANK}")
    print(f"Resolution:      {RESOLUTION}x{RESOLUTION}")
    print(f"Learning rate:   {LEARNING_RATE}")
    print(f"Effective batch: {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print()

    for cls in args.classes:
        cls_dir = os.path.join(args.train_data_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"  {cls:6s}: directory not found")
            continue
        n = len([
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        print(f"  {cls:6s}: {n:5d} images -> {get_max_steps(n):5d} steps")

    print(f"\nTraining {len(args.classes)} classes sequentially...")

    results = {}
    total_start = time.time()
    for cls in args.classes:
        results[cls] = train_lora_for_class(
            cls, args.train_data_dir, args.output_dir, train_script
        )

    total_min = (time.time() - total_start) / 60
    print(f"\n{'=' * 60}")
    print(f"LORA TRAINING COMPLETE — {total_min:.1f} min total")
    print(f"{'=' * 60}")
    for cls, ok in results.items():
        print(f"  [{'OK ' if ok else 'FAIL'}] {cls}")

    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
