"""
DermaDiff - PanDerm ViT-Large Classifier Training Script
Experiment C1: Train PanDerm on HAM10000 + SD 3.5 synthetic images (1x and 2x).

Usage (Google Colab):
    1. Run finetune_lora.py and generate_images.py first
    2. Run this script: !python panderm_classifier.py
    3. Trains C1-1x and C1-2x, saves checkpoints to Drive

GPU: T4 (bs=16, accum=8) or A100 (bs=128)
Estimated time: ~3-4 hours per ratio
"""

import os
import sys
import json
import shutil
import random
import subprocess

import pandas as pd
import numpy as np

# ============================================================
# 1. CONFIGURATION
# ============================================================

# Project paths
PROJECT_ROOT = '/content/drive/MyDrive/DermaDiff'
SHARED = os.path.join(PROJECT_ROOT, 'shared')
NOTEBOOK_DIR = os.path.join(PROJECT_ROOT, 'notebooks/exp_c1')

# Load shared config, splits, label mapping
with open(os.path.join(SHARED, 'config/shared_config.json')) as f:
    CONFIG = json.load(f)
with open(os.path.join(SHARED, 'splits/ham10000_splits.json')) as f:
    SPLITS = json.load(f)
with open(os.path.join(SHARED, 'config/label_mapping.json')) as f:
    label_map = json.load(f)
inv_map = {v: k for k, v in label_map.items()}

# Dataset paths
HAM_IMAGES = os.path.join(PROJECT_ROOT, 'data/ham10000/images')
SYNTHETIC_DIR = os.path.join(NOTEBOOK_DIR, 'synthetic_sd35')

# PanDerm paths
PANDERM_DIR = '/content/PanDerm'
WEIGHT_FILE = os.path.join(SHARED, 'weights/panderm_ll_data6_checkpoint-499.pth')

TARGET_CLASSES = ['mel', 'bcc', 'akiec', 'df', 'vasc']

os.environ['WANDB_MODE'] = 'disabled'


# ============================================================
# 2. GPU AUTO-CONFIGURATION
# ============================================================

def get_batch_config():
    """Auto-configure batch size based on GPU VRAM."""
    import torch
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_mem / 1024**3

    print(f"GPU: {gpu_name}")
    print(f"VRAM: {vram:.1f} GB")

    if vram < 20:  # T4 (16GB)
        batch_size = 16
        update_freq = 8  # effective batch = 128
        print("T4 detected: batch_size=16, update_freq=8 (effective=128)")
    else:  # A100
        batch_size = CONFIG['batch_size']
        update_freq = 1
        print(f"A100 detected: batch_size={batch_size}, update_freq=1")

    return batch_size, update_freq


# ============================================================
# 3. BUILD TRAINING DATA
# ============================================================

def build_base_csv():
    """Build base PanDerm CSV from HAM10000 (same as Exp A)."""
    ham_meta = os.path.join(PROJECT_ROOT, 'data/ham10000/HAM10000_metadata.csv')
    ham_df = pd.read_csv(ham_meta)
    train_ids = set(SPLITS['train'])
    id_col = SPLITS['metadata']['id_column']

    records = []
    for _, row in ham_df.iterrows():
        img_id = row[id_col]
        dx = row['dx']
        label_idx = label_map[dx]

        if img_id in train_ids:
            split = 'train'
        elif img_id in set(SPLITS['val']):
            split = 'val'
        elif img_id in set(SPLITS['test']):
            split = 'test'
        else:
            continue

        records.append({
            'image': f"{img_id}.jpg",
            'label': label_idx,
            'split': split,
        })

    base_df = pd.DataFrame(records)
    print(f"Base CSV: {len(base_df)} rows")
    print(f"  Train: {(base_df['split']=='train').sum()}")
    print(f"  Val:   {(base_df['split']=='val').sum()}")
    print(f"  Test:  {(base_df['split']=='test').sum()}")
    return base_df


def build_synthetic_records(base_df):
    """Build synthetic image records for 1x and 2x ratios."""
    ham_meta = os.path.join(PROJECT_ROOT, 'data/ham10000/HAM10000_metadata.csv')
    ham_df = pd.read_csv(ham_meta)
    train_ids = set(SPLITS['train'])
    id_col = SPLITS['metadata']['id_column']
    ham_train = ham_df[ham_df[id_col].isin(train_ids)]

    random.seed(CONFIG.get('split_seed', 42))

    synth_records_2x = []
    synth_records_1x = []

    for cls in TARGET_CLASSES:
        cls_dir = os.path.join(SYNTHETIC_DIR, cls)
        label_idx = label_map[cls]
        train_count = int((ham_train['dx'] == cls).sum())

        all_synth = sorted([f for f in os.listdir(cls_dir) if f.endswith('.jpg')])

        # 2x: use all
        for fname in all_synth:
            synth_records_2x.append({'image': fname, 'label': label_idx, 'split': 'train'})

        # 1x: subsample to match train count
        subset = random.sample(all_synth, min(train_count, len(all_synth)))
        for fname in subset:
            synth_records_1x.append({'image': fname, 'label': label_idx, 'split': 'train'})

        print(f"  {cls:6s}: 2x={len(all_synth):5d}, 1x={len(subset):4d} (train={train_count})")

    synth_1x_df = pd.DataFrame(synth_records_1x)
    synth_2x_df = pd.DataFrame(synth_records_2x)

    # Combined CSVs
    c1_1x_df = pd.concat([base_df, synth_1x_df], ignore_index=True)
    c1_2x_df = pd.concat([base_df, synth_2x_df], ignore_index=True)

    # Save
    temp_dir = os.path.join(NOTEBOOK_DIR, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    c1_1x_csv = os.path.join(temp_dir, 'ham10000_exp_c1_1x.csv')
    c1_2x_csv = os.path.join(temp_dir, 'ham10000_exp_c1_2x.csv')
    c1_1x_df.to_csv(c1_1x_csv, index=False)
    c1_2x_df.to_csv(c1_2x_csv, index=False)

    print(f"\nC1-1x CSV: {len(c1_1x_df)} rows (train={(c1_1x_df['split']=='train').sum()})")
    print(f"C1-2x CSV: {len(c1_2x_df)} rows (train={(c1_2x_df['split']=='train').sum()})")

    return c1_1x_csv, c1_2x_csv, synth_1x_df, synth_2x_df


def build_image_directories(synth_1x_df, synth_2x_df):
    """Create combined image directories (symlink real + copy synthetic)."""
    c1_1x_images = '/content/exp_c1_1x_images'
    c1_2x_images = '/content/exp_c1_2x_images'

    for img_dir, ratio_label, synth_df in [
        (c1_1x_images, '1x', synth_1x_df),
        (c1_2x_images, '2x', synth_2x_df),
    ]:
        if os.path.exists(img_dir):
            print(f"{ratio_label}: Directory already exists ({len(os.listdir(img_dir))} files)")
            continue

        os.makedirs(img_dir, exist_ok=True)
        print(f"\nBuilding {ratio_label} image directory...")

        # Symlink all HAM10000 images
        ham_count = 0
        for img_file in os.listdir(HAM_IMAGES):
            src = os.path.join(HAM_IMAGES, img_file)
            dst = os.path.join(img_dir, img_file)
            if os.path.isfile(src) and not os.path.exists(dst):
                os.symlink(src, dst)
                ham_count += 1

        # Copy synthetic images
        synth_count = 0
        synth_filenames = set(synth_df['image'].tolist())
        for cls in TARGET_CLASSES:
            cls_dir = os.path.join(SYNTHETIC_DIR, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname in synth_filenames:
                    src = os.path.join(cls_dir, fname)
                    dst = os.path.join(img_dir, fname)
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                        synth_count += 1

        total = len(os.listdir(img_dir))
        print(f"  {ratio_label}: HAM={ham_count} + synth={synth_count} = {total} total files")

    return c1_1x_images, c1_2x_images


# ============================================================
# 4. TRAIN PANDERM
# ============================================================

def train_panderm(csv_path, images_dir, output_dir, drive_dir, exp_name, batch_size, update_freq):
    """Train PanDerm ViT-Large classifier."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(drive_dir, exist_ok=True)

    # Skip if already trained
    if os.path.exists(os.path.join(drive_dir, 'checkpoint-best.pth')):
        print(f"{exp_name} already trained! Checkpoint found on Drive.")
        return True

    cmd = f"""cd {PANDERM_DIR}/classification && \
CUDA_VISIBLE_DEVICES=0 python3 run_class_finetuning.py \
    --model {CONFIG['model_name']} \
    --pretrained_checkpoint {WEIGHT_FILE} \
    --nb_classes {CONFIG['nb_classes']} \
    --batch_size {batch_size} \
    --lr {CONFIG['lr']} \
    --update_freq {update_freq} \
    --warmup_epochs {CONFIG['warmup_epochs']} \
    --epochs {CONFIG['epochs']} \
    --layer_decay {CONFIG['layer_decay']} \
    --drop_path {CONFIG['drop_path']} \
    --weight_decay {CONFIG['weight_decay']} \
    --mixup {CONFIG['mixup']} \
    --cutmix {CONFIG['cutmix']} \
    --weights \
    --sin_pos_emb \
    --no_auto_resume \
    --imagenet_default_mean_and_std \
    --exp_name "{exp_name}" \
    --output_dir {output_dir} \
    --csv_path {csv_path} \
    --root_path "{images_dir}/" \
    --seed {CONFIG['training_seeds'][0]}"""

    print(f"Starting {exp_name} training...")
    exit_code = os.system(cmd)

    if exit_code == 0:
        shutil.copytree(output_dir, drive_dir, dirs_exist_ok=True)
        print(f"{exp_name} results saved to: {drive_dir}")
        return True
    else:
        print(f"{exp_name} training FAILED (exit code {exit_code})")
        return False


# ============================================================
# 5. MAIN
# ============================================================

def main():
    print("DermaDiff - PanDerm ViT-Large Classifier Training")
    print("=" * 60)

    # GPU config
    import torch
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name} ({vram:.1f} GB)")

    if vram < 20:
        batch_size, update_freq = 16, 8
        print("T4 mode: batch_size=16, update_freq=8")
    else:
        batch_size, update_freq = CONFIG['batch_size'], 1
        print(f"A100 mode: batch_size={batch_size}, update_freq=1")

    # Step 1: Build CSVs
    print("\nStep 1: Building training CSVs...")
    base_df = build_base_csv()
    c1_1x_csv, c1_2x_csv, synth_1x_df, synth_2x_df = build_synthetic_records(base_df)

    # Step 2: Build image directories
    print("\nStep 2: Building image directories...")
    c1_1x_images, c1_2x_images = build_image_directories(synth_1x_df, synth_2x_df)

    # Step 3: Train C1-1x
    print("\n" + "=" * 60)
    print("Step 3: Training C1-1x (HAM10000 + 1x SD 3.5 synthetics)")
    print("=" * 60)
    c1_1x_output = '/content/exp_c1_1x_output'
    c1_1x_drive = os.path.join(NOTEBOOK_DIR, 'outputs/exp_c1_1x')
    train_panderm(c1_1x_csv, c1_1x_images, c1_1x_output, c1_1x_drive,
                  "DermaDiff_ExpC1_1x", batch_size, update_freq)

    # Step 4: Train C1-2x
    print("\n" + "=" * 60)
    print("Step 4: Training C1-2x (HAM10000 + 2x SD 3.5 synthetics)")
    print("=" * 60)
    c1_2x_output = '/content/exp_c1_2x_output'
    c1_2x_drive = os.path.join(NOTEBOOK_DIR, 'outputs/exp_c1_2x')
    train_panderm(c1_2x_csv, c1_2x_images, c1_2x_output, c1_2x_drive,
                  "DermaDiff_ExpC1_2x", batch_size, update_freq)

    # Verify checkpoints
    print("\nCHECKPOINTS ON DRIVE:")
    print("=" * 60)
    for name, drive_dir in [('C1-1x', c1_1x_drive), ('C1-2x', c1_2x_drive)]:
        ckpt = os.path.join(drive_dir, 'checkpoint-best.pth')
        if os.path.exists(ckpt):
            size = os.path.getsize(ckpt) / 1024 / 1024
            print(f"  OK   {name}: {size:.1f} MB")
        else:
            print(f"  MISS {name}: NOT FOUND")


if __name__ == '__main__':
    main()
