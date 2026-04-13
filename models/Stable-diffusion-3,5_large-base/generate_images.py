"""
DermaDiff - SD 3.5 Large Synthetic Image Generation Script
Experiment C1: Generate synthetic dermoscopic images using trained LoRA adapters.

Usage (Google Colab):
    1. Run finetune_lora.py first (all 5 LoRA weights must exist)
    2. Run this script: !python generate_images.py
    3. Crash-safe: re-run skips already generated images

GPU: A100 40GB required
Estimated time: ~3 hours total
"""

import os
import sys
import json
import time
import math
import random

import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusion3Pipeline

# ============================================================
# 1. CONFIGURATION
# ============================================================

# Project paths
PROJECT_ROOT = '/content/drive/MyDrive/DermaDiff'
SHARED = os.path.join(PROJECT_ROOT, 'shared')
NOTEBOOK_DIR = os.path.join(PROJECT_ROOT, 'notebooks/exp_c1')

# Load shared config & splits
with open(os.path.join(SHARED, 'config/shared_config.json')) as f:
    CONFIG = json.load(f)
with open(os.path.join(SHARED, 'splits/ham10000_splits.json')) as f:
    SPLITS = json.load(f)
with open(os.path.join(SHARED, 'config/label_mapping.json')) as f:
    label_map = json.load(f)

# LoRA weights from finetune_lora.py
LORA_DRIVE_BASE = os.path.join(NOTEBOOK_DIR, 'lora_weights')

TARGET_CLASSES = ['mel', 'bcc', 'akiec', 'df', 'vasc']

# SD 3.5 Large generation settings
SD35_MODEL = 'stabilityai/stable-diffusion-3.5-large'
NUM_INFERENCE_STEPS = 28
GUIDANCE_SCALE = 4.5
RESOLUTION = 1024
MAX_RATIO = 2       # Generate 2x, subsample for 1x later
BATCH_SIZE = 2       # Safe for A100 at 1024x1024
BASE_SEED = 42
MAX_SEQ_LENGTH = 256

# Prompt variants per class (5 per class, rotated for diversity)
CLASS_PROMPTS = {
    'mel': [
        'a dermoscopic photograph of melanoma with irregular pigment network, multiple colors including brown black and blue-gray, asymmetric borders, on light skin',
        'a dermoscopic close-up of melanoma skin lesion showing atypical pigment network with blue-white veil, irregular dots and globules, on tan skin background',
        'a clinical dermoscopic image of melanoma with irregular streaks, regression structures, multiple shades of brown and dark pigmentation, scar-like depigmentation',
        'a dermoscopic photograph of superficial spreading melanoma with asymmetric pigmentation, dark brown to black blotches, irregular border, on fair skin',
        'a dermoscopic image of early melanoma showing atypical network with wide irregular meshes, brown and blue-gray colors, structureless areas',
    ],
    'bcc': [
        'a dermoscopic photograph of basal cell carcinoma with arborizing telangiectasia vessels, blue-gray ovoid nests, translucent pearly background on light skin',
        'a dermoscopic close-up of pigmented basal cell carcinoma showing leaf-like areas, blue-gray globules, branching tree-like vessels, ulceration',
        'a clinical dermoscopic image of nodular basal cell carcinoma with bright red arborizing vessels, blue-gray structures, shiny white areas on pale skin',
        'a dermoscopic photograph of basal cell carcinoma with spoke-wheel structures, multiple blue-gray dots, fine telangiectasia, on skin colored background',
        'a dermoscopic image of BCC showing large blue-gray ovoid nests, arborizing vessels with fine branches, absence of pigment network',
    ],
    'akiec': [
        'a dermoscopic photograph of actinic keratosis with strawberry pattern, red pseudonetwork, white-yellow follicular plugs, surface scale on sun-damaged skin',
        'a dermoscopic close-up of actinic keratosis showing erythematous background, prominent yellowish hair follicles with white halo, fine wavy vessels',
        'a clinical dermoscopic image of actinic keratosis with white opaque scales on reddish background, rosettes, targetoid follicular openings on aged skin',
        'a dermoscopic photograph of keratosis showing rough scaly surface, red erythema, prominent follicular openings surrounded by white circles, on fair skin',
        'a dermoscopic image of actinic keratosis with fine linear vessels, white-yellow keratotic surface, erythematous pseudonetwork pattern',
    ],
    'df': [
        'a dermoscopic photograph of dermatofibroma with central white scar-like patch surrounded by delicate peripheral brown pigment network on skin',
        'a dermoscopic close-up of dermatofibroma showing central white fibrotic area with peripheral light brown reticular network that fades into surrounding skin',
        'a clinical dermoscopic image of dermatofibroma with bright white central patch, dotted vessels, delicate tan peripheral network on light-to-medium skin tone',
        'a dermoscopic photograph of dermatofibroma with homogeneous white center and ring-like brown structures at periphery, smooth dome-shaped on leg skin',
        'a dermoscopic image of dermatofibroma showing crystalline white structures in center with surrounding fine pigment network, brown to tan coloration',
    ],
    'vasc': [
        'a dermoscopic photograph of vascular lesion with well-demarcated red and blue-red lacunae of varying sizes clustered against reddish background',
        'a dermoscopic close-up of cherry angioma showing multiple round red lacunae separated by whitish septa on skin surface',
        'a clinical dermoscopic image of vascular skin lesion with dark red to purple lacunae, reddish-blue homogeneous background, dilated blood vessels',
        'a dermoscopic photograph of angioma with clustered red-blue lacunae varying in size, thrombosed dark areas, against red homogeneous background',
        'a dermoscopic image of vascular lesion showing well-defined round to oval red lacunae with reddish-brown coloration and whitish fibrous septa',
    ],
}

# Output paths
LOCAL_OUTPUT = '/content/synthetic_images'
DRIVE_OUTPUT = os.path.join(NOTEBOOK_DIR, 'synthetic_sd35')


# ============================================================
# 2. COMPUTE GENERATION TARGETS
# ============================================================

def get_train_counts():
    """Get per-class train counts from HAM10000."""
    ham_meta = os.path.join(PROJECT_ROOT, 'data/ham10000/HAM10000_metadata.csv')
    ham_df = pd.read_csv(ham_meta)
    train_ids = set(SPLITS['train'])
    id_col = SPLITS['metadata']['id_column']
    ham_train = ham_df[ham_df[id_col].isin(train_ids)]

    counts = {}
    for cls in TARGET_CLASSES:
        counts[cls] = int((ham_train['dx'] == cls).sum())
    return counts


# ============================================================
# 3. GENERATION
# ============================================================

def generate_for_class(pipe, cls_name, num_images):
    """Generate synthetic images for one class using its LoRA."""
    cls_local = os.path.join(LOCAL_OUTPUT, cls_name)
    cls_drive = os.path.join(DRIVE_OUTPUT, cls_name)
    os.makedirs(cls_local, exist_ok=True)
    os.makedirs(cls_drive, exist_ok=True)

    # Check if already generated on Drive (crash-safe)
    existing = len([f for f in os.listdir(cls_drive) if f.endswith('.jpg')]) if os.path.exists(cls_drive) else 0
    if existing >= num_images:
        print(f"  SKIP {cls_name}: {existing} images already on Drive (need {num_images})")
        return existing

    # Load LoRA for this class
    lora_path = os.path.join(LORA_DRIVE_BASE, cls_name)
    pipe.load_lora_weights(lora_path)

    prompts = CLASS_PROMPTS[cls_name]
    generated = 0
    num_batches = math.ceil(num_images / BATCH_SIZE)

    print(f"  Generating {num_images} images for {cls_name} ({num_batches} batches)...")

    for batch_idx in tqdm(range(num_batches), desc=f"  {cls_name}"):
        batch_count = min(BATCH_SIZE, num_images - generated)

        # Rotate through prompt variants
        batch_prompts = [prompts[(batch_idx * BATCH_SIZE + i) % len(prompts)] for i in range(batch_count)]

        # Unique seed per batch
        seed = BASE_SEED + batch_idx * BATCH_SIZE
        generator = torch.Generator(device='cuda').manual_seed(seed)

        with torch.no_grad():
            result = pipe(
                prompt=batch_prompts,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                height=RESOLUTION,
                width=RESOLUTION,
                max_sequence_length=MAX_SEQ_LENGTH,
                generator=generator,
            )

        for i, img in enumerate(result.images):
            img_idx = generated + i
            filename = f"sd35_{cls_name}_{img_idx:04d}.jpg"

            local_path = os.path.join(cls_local, filename)
            img.save(local_path, quality=95)

            # Save to Drive incrementally (crash-safe)
            drive_path = os.path.join(cls_drive, filename)
            img.save(drive_path, quality=95)

        generated += batch_count

        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()

    # Unload LoRA before next class
    pipe.unload_lora_weights()
    torch.cuda.empty_cache()

    return generated


def verify_generation(train_counts):
    """Verify all synthetic images exist on Drive."""
    print("\nSYNTHETIC IMAGES ON DRIVE:")
    print("=" * 60)

    all_ok = True
    total = 0

    for cls in TARGET_CLASSES:
        cls_dir = os.path.join(DRIVE_OUTPUT, cls)
        target = train_counts[cls] * MAX_RATIO

        if os.path.exists(cls_dir):
            count = len([f for f in os.listdir(cls_dir) if f.endswith('.jpg')])
        else:
            count = 0

        total += count
        status = 'OK' if count >= target else 'LOW'
        if count < target:
            all_ok = False
        print(f"  [{status:3s}] {cls:6s}: {count:5d} / {target:5d} images")

    print(f"\n  Total: {total} synthetic images")
    print(f"  Location: {DRIVE_OUTPUT}")

    return all_ok


# ============================================================
# 4. MAIN
# ============================================================

def main():
    print("DermaDiff - SD 3.5 Large Synthetic Image Generation")
    print("=" * 60)
    print(f"Model:      {SD35_MODEL}")
    print(f"Steps:      {NUM_INFERENCE_STEPS}")
    print(f"Guidance:   {GUIDANCE_SCALE}")
    print(f"Resolution: {RESOLUTION}x{RESOLUTION}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Ratio:      {MAX_RATIO}x")
    print()

    # Get generation targets
    train_counts = get_train_counts()
    print("Generation targets:")
    total_images = 0
    for cls in TARGET_CLASSES:
        n_gen = train_counts[cls] * MAX_RATIO
        total_images += n_gen
        print(f"  {cls:6s}: {train_counts[cls]:4d} train x {MAX_RATIO} = {n_gen:5d} images")
    print(f"  Total: {total_images}")

    # Verify LoRA weights
    print("\nChecking LoRA weights...")
    all_ok = True
    for cls in TARGET_CLASSES:
        lora_file = os.path.join(LORA_DRIVE_BASE, cls, 'pytorch_lora_weights.safetensors')
        exists = os.path.exists(lora_file)
        size = f"{os.path.getsize(lora_file)/1024/1024:.1f} MB" if exists else "MISSING"
        status = 'OK' if exists else 'FAIL'
        print(f"  [{status}] {cls:6s}: {size}")
        if not exists:
            all_ok = False
    assert all_ok, "Missing LoRA weights! Run finetune_lora.py first."

    # Load pipeline
    print("\nLoading SD 3.5 Large base model...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        SD35_MODEL,
        torch_dtype=torch.bfloat16,
        token=True,
        use_safetensors=True,
    ).to('cuda')
    print("Pipeline loaded.")

    # Generate all classes
    os.makedirs(LOCAL_OUTPUT, exist_ok=True)
    os.makedirs(DRIVE_OUTPUT, exist_ok=True)

    results = {}
    total_start = time.time()

    for cls in TARGET_CLASSES:
        num_to_gen = train_counts[cls] * MAX_RATIO
        start = time.time()

        print(f"\n{'='*60}")
        print(f"  {cls}: generating {num_to_gen} images (2x of {train_counts[cls]})")
        print(f"{'='*60}")

        count = generate_for_class(pipe, cls, num_to_gen)
        elapsed = (time.time() - start) / 60

        results[cls] = count
        print(f"  Done: {count} images in {elapsed:.1f} min")

    total_time = (time.time() - total_start) / 60

    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE - Total time: {total_time:.1f} min")
    print("=" * 60)
    for cls, count in results.items():
        target = train_counts[cls] * MAX_RATIO
        status = 'OK' if count >= target else 'LOW'
        print(f"  [{status:3s}] {cls:6s}: {count}/{target}")

    # Cleanup
    del pipe
    torch.cuda.empty_cache()

    # Verify
    verify_generation(train_counts)


if __name__ == '__main__':
    main()
