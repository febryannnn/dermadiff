#!/usr/bin/env python3
"""
DermaDiff — Phase 1: SD 3.5 Large LoRA Fine-tuning
====================================================

Fine-tunes Stable Diffusion 3.5 Large with LoRA adapters on dermoscopic skin
lesion images, producing one LoRA per minority class (mel, bcc, akiec, df, vasc).

This script wraps HuggingFace diffusers'
`examples/dreambooth/train_dreambooth_lora_sd3.py` — the official SD3 DreamBooth
LoRA training script. SD 3.5 Large uses the MMDiT (Multimodal Diffusion
Transformer) architecture with rectified flow matching, requiring bf16 precision
and gradient checkpointing for stable training on A100 GPUs.

Usage
-----
    python finetune_lora.py \
        --train_data_dir /path/to/training_images \
        --output_dir /path/to/lora_weights \
        --diffusers_dir /path/to/diffusers_repo

Each class subdirectory under --train_data_dir must contain dermoscopic images
(e.g. train_data_dir/mel/*.jpg, train_data_dir/bcc/*.jpg, ...).

Requirements
------------
    pip install git+https://github.com/huggingface/diffusers.git
    pip install transformers accelerate "peft>=0.17.0" bitsandbytes
    pip install safetensors sentencepiece protobuf
    git clone https://github.com/huggingface/diffusers.git

GPU: A100 40GB required
HuggingFace: Must accept license at huggingface.co/stabilityai/stable-diffusion-3.5-large
"""

import argparse
import os
import subprocess
import sys
import time

SD35_MODEL = "stabilityai/stable-diffusion-3.5-large"
LORA_RANK = 64
LEARNING_RATE = 4e-4
RESOLUTION = 1024
TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4
LR_SCHEDULER = "constant"
LR_WARMUP_STEPS = 0
SEED = 42
EFFECTIVE_BATCH = TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION

TARGET_CLASSES = ["mel", "bcc", "akiec", "df", "vasc"]

CLASS_PROMPTS = {
    "mel":   "a dermoscopic photograph of a melanoma skin lesion",
    "bcc":   "a dermoscopic photograph of a basal cell carcinoma skin lesion",
    "akiec": "a dermoscopic photograph of an actinic keratosis skin lesion",
    "df":    "a dermoscopic photograph of a dermatofibroma skin lesion",
    "vasc":  "a dermoscopic photograph of a vascular skin lesion",
}


def get_max_steps(num_images: int) -> int:
    steps_per_epoch = max(1, num_images // EFFECTIVE_BATCH)
    if num_images > 2000:
        target_epochs = 3
    elif num_images > 500:
        target_epochs = 5
    elif num_images > 200:
        target_epochs = 10
    else:
        target_epochs = 15
    return max(500, steps_per_epoch * target_epochs)


def train_lora_for_class(cls_name, train_data_dir, output_dir, diffusers_dir):
    cls_data_dir = os.path.join(train_data_dir, cls_name)
    cls_output_dir = os.path.join(output_dir, cls_name)
    os.makedirs(cls_output_dir, exist_ok=True)

    weights_file = os.path.join(cls_output_dir, "pytorch_lora_weights.safetensors")
    if os.path.exists(weights_file):
        size_mb = os.path.getsize(weights_file) / 1024 / 1024
        print(f"  SKIP {cls_name}: already trained ({size_mb:.1f} MB)")
        return True

    if not os.path.isdir(cls_data_dir):
        print(f"  SKIP {cls_name}: directory not found at {cls_data_dir}")
        return False

    n_images = len([f for f in os.listdir(cls_data_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if n_images == 0:
        print(f"  SKIP {cls_name}: no images in {cls_data_dir}")
        return False

    max_steps = get_max_steps(n_images)
    caption = CLASS_PROMPTS[cls_name]
    train_script = os.path.join(diffusers_dir, "examples/dreambooth/train_dreambooth_lora_sd3.py")

    if not os.path.exists(train_script):
        print(f"  ERROR: training script not found at {train_script}")
        return False

    print(f"\n{'=' * 60}")
    print(f"  Training LoRA for: {cls_name}")
    print(f"  Images: {n_images} | Steps: {max_steps}")
    print(f"  Prompt: {caption}")
    print(f"{'=' * 60}")

    cmd = [
        "accelerate", "launch", train_script,
        f"--pretrained_model_name_or_path={SD35_MODEL}",
        f"--instance_data_dir={cls_data_dir}",
        f"--output_dir={cls_output_dir}",
        "--mixed_precision=bf16",
        f"--instance_prompt={caption}",
        f"--resolution={RESOLUTION}",
        f"--train_batch_size={TRAIN_BATCH_SIZE}",
        f"--gradient_accumulation_steps={GRADIENT_ACCUMULATION}",
        f"--learning_rate={LEARNING_RATE}",
        f"--lr_scheduler={LR_SCHEDULER}",
        f"--lr_warmup_steps={LR_WARMUP_STEPS}",
        f"--max_train_steps={max_steps}",
        f"--rank={LORA_RANK}",
        f"--seed={SEED}",
        "--gradient_checkpointing",
        "--checkpointing_steps=500",
        "--weighting_scheme=logit_normal",
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
    parser = argparse.ArgumentParser(description="Fine-tune SD 3.5 Large with LoRA on dermoscopic images")
    parser.add_argument("--train_data_dir", required=True, help="Root dir with per-class image subdirectories")
    parser.add_argument("--output_dir", required=True, help="Output directory for LoRA weights")
    parser.add_argument("--diffusers_dir", required=True, help="Path to cloned huggingface/diffusers repo")
    parser.add_argument("--classes", nargs="+", default=TARGET_CLASSES, help=f"Classes to train (default: {' '.join(TARGET_CLASSES)})")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("SD 3.5 LARGE LoRA TRAINING PLAN")
    print("=" * 60)
    print(f"Base model:      {SD35_MODEL}")
    print(f"Architecture:    MMDiT (Rectified Flow Transformer)")
    print(f"LoRA rank:       {LORA_RANK}")
    print(f"Resolution:      {RESOLUTION}x{RESOLUTION}")
    print(f"Learning rate:   {LEARNING_RATE}")
    print(f"Effective batch: {EFFECTIVE_BATCH}")
    print(f"Precision:       bf16")
    print()

    for cls in args.classes:
        cls_dir = os.path.join(args.train_data_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"  {cls:6s}: directory not found")
            continue
        n = len([f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        print(f"  {cls:6s}: {n:5d} images -> {get_max_steps(n):5d} steps")

    print(f"\nTraining {len(args.classes)} classes sequentially...")

    results = {}
    total_start = time.time()
    for cls in args.classes:
        results[cls] = train_lora_for_class(cls, args.train_data_dir, args.output_dir, args.diffusers_dir)

    total_min = (time.time() - total_start) / 60
    print(f"\n{'=' * 60}")
    print(f"LoRA TRAINING COMPLETE — {total_min:.1f} min total")
    print(f"{'=' * 60}")
    for cls, ok in results.items():
        print(f"  [{'OK ' if ok else 'FAIL'}] {cls}")

    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()