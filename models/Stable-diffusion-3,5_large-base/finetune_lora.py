"""
DermaDiff - SD 3.5 Large LoRA Fine-Tuning Script
Experiment C1: Fine-tune one LoRA adapter per minority class using
the official HuggingFace train_dreambooth_lora_sd3.py script.

Usage (Google Colab):
    1. Mount Drive, install dependencies
    2. Run this script: !python finetune_lora.py
    3. Script auto-skips already-trained classes (crash-safe)

GPU: A100 40GB required
Estimated time: ~1-2 hours per class
"""

import os
import sys
import json
import shutil
import time
import subprocess

import pandas as pd
import numpy as np

# ============================================================
# 1. CONFIGURATION
# ============================================================

# Project paths (Google Drive)
PROJECT_ROOT = '/content/drive/MyDrive/DermaDiff'
SHARED = os.path.join(PROJECT_ROOT, 'shared')
NOTEBOOK_DIR = os.path.join(PROJECT_ROOT, 'notebooks/exp_c1')
os.makedirs(NOTEBOOK_DIR, exist_ok=True)

# Load shared config, splits, label mapping
with open(os.path.join(SHARED, 'config/shared_config.json')) as f:
    CONFIG = json.load(f)
with open(os.path.join(SHARED, 'splits/ham10000_splits.json')) as f:
    SPLITS = json.load(f)
with open(os.path.join(SHARED, 'config/label_mapping.json')) as f:
    label_map = json.load(f)

# Dataset paths
HAM_IMAGES = os.path.join(PROJECT_ROOT, 'data/ham10000/images')
HAM_CLASSIFIED = os.path.join(PROJECT_ROOT, 'data/ham10000/images_classified')
ISIC_IMAGES = os.path.join(PROJECT_ROOT, 'data/isic2019/images')
LONG_EXTRA = os.path.join(SHARED, 'diffusion_extra')

# Target minority classes
TARGET_CLASSES = ['mel', 'bcc', 'akiec', 'df', 'vasc']

# Diffusers training script
DIFFUSERS_DIR = '/content/diffusers'
TRAIN_SCRIPT = os.path.join(DIFFUSERS_DIR, 'examples/dreambooth/train_dreambooth_lora_sd3.py')

# SD 3.5 Large LoRA training configuration
SD35_MODEL = 'stabilityai/stable-diffusion-3.5-large'
LORA_RANK = 64
LEARNING_RATE = 4e-4
RESOLUTION = 1024
TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4
LR_SCHEDULER = 'constant'
LR_WARMUP_STEPS = 0
SEED = 42
EFFECTIVE_BATCH = TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION  # 4

# Instance prompts per class (DreamBooth)
CLASS_PROMPTS = {
    'mel':   'a dermoscopic photograph of a melanoma skin lesion',
    'bcc':   'a dermoscopic photograph of a basal cell carcinoma skin lesion',
    'akiec': 'a dermoscopic photograph of an actinic keratosis skin lesion',
    'df':    'a dermoscopic photograph of a dermatofibroma skin lesion',
    'vasc':  'a dermoscopic photograph of a vascular skin lesion',
}

# ISIC 2019 folder name mapping
ISIC_CLASS_MAP = {
    'mel':   'MEL (malignant)',
    'bcc':   'BCC (malignant)',
    'akiec': 'AK (pre-malignant)',
    'df':    'DF (benign)',
    'vasc':  'VASC (benign)',
}

# Output directories
LOCAL_DATA = '/content/lora_data'
LORA_OUTPUT_BASE = '/content/lora_weights'
LORA_DRIVE_BASE = os.path.join(NOTEBOOK_DIR, 'lora_weights')

os.environ['WANDB_MODE'] = 'disabled'


# ============================================================
# 2. ADAPTIVE STEPS COMPUTATION
# ============================================================

def get_max_steps(num_images):
    """Compute training steps based on dataset size.
    Large datasets: fewer epochs (already diverse).
    Small datasets: more epochs (need repetition).
    """
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


# ============================================================
# 3. DATA COLLECTION
# ============================================================

def collect_training_data():
    """Collect images from HAM10000, ISIC 2019, and Longitudinal
    into per-class local directories for fast I/O during training."""

    # Load HAM10000 metadata and filter to train split
    ham_meta = os.path.join(PROJECT_ROOT, 'data/ham10000/HAM10000_metadata.csv')
    ham_df = pd.read_csv(ham_meta)
    train_ids = set(SPLITS['train'])
    id_col = SPLITS['metadata']['id_column']
    ham_train = ham_df[ham_df[id_col].isin(train_ids)].copy()

    print(f"HAM10000 train images: {len(ham_train)}")

    for cls in TARGET_CLASSES:
        cls_dir = os.path.join(LOCAL_DATA, cls)
        os.makedirs(cls_dir, exist_ok=True)

        # Source 1: HAM10000 train split ONLY
        cls_train_ids = ham_train[ham_train['dx'] == cls][id_col].tolist()
        ham_count = 0
        for img_id in cls_train_ids:
            src = os.path.join(HAM_IMAGES, f"{img_id}.jpg")
            dst = os.path.join(cls_dir, f"ham_{img_id}.jpg")
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
                ham_count += 1

        # Source 2: ISIC 2019
        isic_folder = ISIC_CLASS_MAP.get(cls)
        isic_count = 0
        if isic_folder:
            isic_cls_dir = os.path.join(ISIC_IMAGES, isic_folder)
            if os.path.isdir(isic_cls_dir):
                for img_file in os.listdir(isic_cls_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        src = os.path.join(isic_cls_dir, img_file)
                        dst = os.path.join(cls_dir, f"isic_{img_file}")
                        if not os.path.exists(dst):
                            shutil.copy2(src, dst)
                            isic_count += 1

        # Source 3: Longitudinal extras
        long_cls_dir = os.path.join(LONG_EXTRA, cls)
        long_count = 0
        if os.path.isdir(long_cls_dir):
            for img_file in os.listdir(long_cls_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src = os.path.join(long_cls_dir, img_file)
                    dst = os.path.join(cls_dir, f"long_{img_file}")
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                        long_count += 1

        total = len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"  {cls:6s}: HAM={ham_count:4d} + ISIC={isic_count:4d} + Long={long_count:3d} = {total:5d} total")

    print(f"\nAll images collected to {LOCAL_DATA}")


# ============================================================
# 4. LORA TRAINING
# ============================================================

def train_lora_for_class(cls_name):
    """Train SD 3.5 Large LoRA for one class using official script."""
    cls_dir = os.path.join(LOCAL_DATA, cls_name)
    output_dir = os.path.join(LORA_OUTPUT_BASE, cls_name)
    drive_dir = os.path.join(LORA_DRIVE_BASE, cls_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(drive_dir, exist_ok=True)

    # Skip if already trained
    lora_file = os.path.join(drive_dir, 'pytorch_lora_weights.safetensors')
    if os.path.exists(lora_file):
        size_mb = os.path.getsize(lora_file) / 1024 / 1024
        print(f"  SKIP {cls_name}: LoRA weights already exist on Drive ({size_mb:.1f} MB)")
        return True

    n_images = len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    max_steps = get_max_steps(n_images)
    caption = CLASS_PROMPTS[cls_name]

    print(f"\n{'='*60}")
    print(f"  Training LoRA for: {cls_name}")
    print(f"  Images: {n_images} | Steps: {max_steps} | Prompt: {caption[:50]}...")
    print(f"{'='*60}")

    start = time.time()

    cmd = [
        "accelerate", "launch", TRAIN_SCRIPT,
        f"--pretrained_model_name_or_path={SD35_MODEL}",
        f"--instance_data_dir={cls_dir}",
        f"--output_dir={output_dir}",
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

    result = subprocess.run(cmd)
    exit_code = result.returncode
    elapsed = (time.time() - start) / 60

    if exit_code == 0:
        local_lora = os.path.join(output_dir, 'pytorch_lora_weights.safetensors')
        if os.path.exists(local_lora):
            shutil.copy2(local_lora, drive_dir)
            size_mb = os.path.getsize(local_lora) / 1024 / 1024
            print(f"  SUCCESS: {cls_name} LoRA saved ({size_mb:.1f} MB) in {elapsed:.1f} min")
            return True
        else:
            print(f"  WARNING: Training finished but no weights file found!")
            print(f"  Check output dir: {output_dir}")
            for f in os.listdir(output_dir):
                print(f"    {f}")
            return False
    else:
        print(f"  FAILED: {cls_name} training failed (exit code {exit_code}) after {elapsed:.1f} min")
        return False


def verify_weights():
    """Verify all LoRA weights exist on Drive."""
    print("\nLoRA WEIGHTS ON DRIVE:")
    print("=" * 60)
    all_ok = True
    for cls in TARGET_CLASSES:
        lora_file = os.path.join(LORA_DRIVE_BASE, cls, 'pytorch_lora_weights.safetensors')
        if os.path.exists(lora_file):
            size_mb = os.path.getsize(lora_file) / 1024 / 1024
            print(f"  OK   {cls:6s}: {size_mb:.1f} MB")
        else:
            print(f"  MISS {cls:6s}: NOT FOUND")
            all_ok = False

    if all_ok:
        print("\nAll 5 LoRA weights saved.")
        print(f"Location: {LORA_DRIVE_BASE}")
    else:
        print("\nWARNING: Some LoRA weights are missing!")

    return all_ok


# ============================================================
# 5. MAIN
# ============================================================

def main():
    print("DermaDiff - SD 3.5 Large LoRA Fine-Tuning")
    print("=" * 60)
    print(f"Model:          {SD35_MODEL}")
    print(f"LoRA rank:      {LORA_RANK}")
    print(f"Resolution:     {RESOLUTION}x{RESOLUTION}")
    print(f"Learning rate:  {LEARNING_RATE}")
    print(f"Effective batch: {EFFECTIVE_BATCH}")
    print(f"Precision:      bf16")
    print()

    # Step 1: Collect training data
    print("Step 1: Collecting training data...")
    collect_training_data()

    # Step 2: Print training plan
    print("\nStep 2: Training plan")
    total_steps = 0
    for cls in TARGET_CLASSES:
        cls_dir = os.path.join(LOCAL_DATA, cls)
        n_images = len([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        steps = get_max_steps(n_images)
        total_steps += steps
        print(f"  {cls:6s}: {n_images:5d} images -> {steps:5d} steps")
    print(f"  Total: {total_steps} steps")

    # Step 3: Train all classes
    print("\nStep 3: Training LoRA per class...")
    results = {}
    total_start = time.time()

    for cls in TARGET_CLASSES:
        success = train_lora_for_class(cls)
        results[cls] = 'OK' if success else 'FAILED'

    total_time = (time.time() - total_start) / 60

    print(f"\n{'='*60}")
    print(f"LoRA TRAINING COMPLETE - Total time: {total_time:.1f} min")
    print("=" * 60)
    for cls, status in results.items():
        icon = 'OK' if status == 'OK' else 'FAIL'
        print(f"  [{icon:4s}] {cls}")

    # Step 4: Verify
    verify_weights()


if __name__ == '__main__':
    main()
