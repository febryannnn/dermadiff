#!/usr/bin/env python3
"""
DermaDiff — Phase 1 (DoRA): SDXL DoRA Fine-tuning
====================================================

Fine-tunes Stable Diffusion XL with DoRA (Weight-Decomposed Low-Rank
Adaptation, Liu et al. ICML 2024) adapters on dermoscopic skin lesion images,
producing one DoRA adapter per minority class (mel, bcc, akiec, df, vasc).

DoRA decomposes the pretrained weights into magnitude and direction vectors,
then applies LoRA only to the directional component. It typically matches or
beats standard LoRA at the same rank, and can match full fine-tuning quality
at much lower parameter counts.

This script wraps HuggingFace diffusers'
`examples/dreambooth/train_dreambooth_lora_sdxl.py` with the `--use_dora` flag.
NOTE: The DreamBooth script is used because it's the ONLY diffusers SDXL LoRA
training script that supports `--use_dora` — the `text_to_image_lora_sdxl.py`
script used for the LoRA variant does NOT support DoRA.

Hyperparameters follow NVIDIA DoRA paper recommendations:
  - rank = 8 (half of LoRA rank 16; paper shows comparable results)
  - lr = 5e-5 (lower than LoRA's 1e-4)
  - lora_dropout = 0.05
  - cosine LR scheduler

The bundled diffusers v0.37.1 DreamBooth script exposes `--lora_dropout` as
a first-class CLI argument, so we pass it directly — no source patching
needed.

The published DermaDiff C3-DoRA results (Macro F1 = 0.8471, matching C4/SD 3.5
Large with a 3x smaller base model) were produced using this exact training
script and configuration.

Usage
-----
    python fine_tuned_DoRA.py \\
        --train_data_dir /path/to/training_images \\
        --output_dir "/path/to/LoRA Weights"

Each class subdirectory under --train_data_dir must contain dermoscopic images
for that class (e.g. train_data_dir/mel/*.jpg, train_data_dir/bcc/*.jpg, ...).
"""

import argparse
import os
import subprocess
import sys
import time

# ────────────────────────────────────────────────────────────────────────
# config (hardcoded — change here, not via CLI)
# ────────────────────────────────────────────────────────────────────────

SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_FIX = "madebyollin/sdxl-vae-fp16-fix"

# DoRA hyperparameters follow the NVIDIA DoRA paper (Liu et al., ICML 2024)
# recommendations: lower rank, lower learning rate, nonzero dropout, cosine
# scheduler. These produced the published C3-DoRA result (Macro F1 = 0.8471,
# matching C4/SD 3.5 Large with a 3x smaller base model).
LORA_RANK = 8              # DoRA paper: half the LoRA rank gives comparable results
LEARNING_RATE = 5e-5       # NVIDIA: "slightly lower lr than LoRA"
LORA_DROPOUT = 0.05        # NVIDIA: "experiment with varying dropout ratios"
RESOLUTION = 1024          # Native SDXL resolution
TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4
LR_SCHEDULER = "cosine"    # smoother convergence than constant
LR_WARMUP_STEPS = 100
SEED = 42

TARGET_CLASSES = ["mel", "bcc", "akiec", "df", "vasc"]

# class-specific instance prompts for DreamBooth fine-tuning.
# unlike LoRA's text_to_image script (which uses metadata.jsonl), DreamBooth
# uses a single --instance_prompt per training run. every image in the class
# implicitly shares this prompt.
CLASS_PROMPTS = {
    "mel":   "a dermoscopic photograph of a melanoma skin lesion",
    "bcc":   "a dermoscopic photograph of a basal cell carcinoma skin lesion",
    "akiec": "a dermoscopic photograph of an actinic keratosis skin lesion",
    "df":    "a dermoscopic photograph of a dermatofibroma skin lesion",
    "vasc":  "a dermoscopic photograph of a vascular skin lesion",
}


def get_max_steps(num_images):
    """Adaptive step count based on dataset size.

    Same as the LoRA variant: small classes (<150 images) get 1500 steps
    (~6 epochs), medium (<400) get 1000 (~3 epochs), large get 800 (~2 epochs).
    """
    if num_images < 150:
        return 1500   # ~6 epochs for tiny classes (df ~94, vasc ~99)
    elif num_images < 400:
        return 1000   # ~3 epochs for medium classes (akiec ~229)
    else:
        return 800    # ~2 epochs for large classes (mel ~779+, bcc ~360+)


def resolve_train_script(diffusers_dir):
    """Locate the DreamBooth+DoRA training script.

    Priority order:
    1. If --diffusers_dir was provided → use that clone's examples/ folder
    2. Otherwise → use the bundled script next to this wrapper
       (models/stable-diffusion-xl-base-dora/train_dreambooth_lora_sdxl.py)
    3. Return None if neither is found.

    The bundled script is sourced from diffusers v0.37.1 and is pinned as
    part of the repo for full reproducibility. Pass --diffusers_dir only if
    you want to override with a newer/different version.
    """
    if diffusers_dir:
        candidate = os.path.join(
            diffusers_dir, "examples/dreambooth/train_dreambooth_lora_sdxl.py"
        )
        if os.path.exists(candidate):
            return candidate
        print(f"  WARNING: --diffusers_dir was given but no script at {candidate}")
        return None

    # bundled script lives next to this wrapper file
    bundled = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "train_dreambooth_lora_sdxl.py",
    )
    if os.path.exists(bundled):
        return bundled

    return None


def train_dora_for_class(
    cls_name: str,
    train_data_dir: str,
    output_dir: str,
    train_script: str,
) -> bool:
    """Train one SDXL DoRA adapter for a single skin lesion class."""

    cls_data_dir = os.path.join(train_data_dir, cls_name)
    cls_output_dir = os.path.join(output_dir, f"lora_{cls_name}_final")
    os.makedirs(cls_output_dir, exist_ok=True)

    # skip if already trained
    weights_file = os.path.join(cls_output_dir, "pytorch_lora_weights.safetensors")
    if os.path.exists(weights_file):
        size_mb = os.path.getsize(weights_file) / 1024 / 1024
        print(f"  SKIP {cls_name}: already trained ({size_mb:.1f} MB)")
        return True

    if not os.path.isdir(cls_data_dir):
        print(f"  SKIP {cls_name}: directory not found at {cls_data_dir}")
        return False

    n_images = len([
        f for f in os.listdir(cls_data_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    if n_images == 0:
        print(f"  SKIP {cls_name}: no images in {cls_data_dir}")
        return False

    max_steps = get_max_steps(n_images)
    prompt = CLASS_PROMPTS[cls_name]

    print(f"\n{'=' * 60}")
    print(f"  Training DoRA for: {cls_name}")
    print(f"  Images: {n_images} | Steps: {max_steps}")
    print(f"  Prompt: {prompt}")
    print(f"  Script: train_dreambooth_lora_sdxl.py --use_dora")
    print(f"{'=' * 60}")

    cmd = [
        "accelerate", "launch", train_script,
        f"--pretrained_model_name_or_path={SDXL_MODEL}",
        f"--pretrained_vae_model_name_or_path={VAE_FIX}",
        f"--instance_data_dir={cls_data_dir}",
        f"--instance_prompt={prompt}",
        f"--output_dir={cls_output_dir}",
        f"--resolution={RESOLUTION}",
        f"--train_batch_size={TRAIN_BATCH_SIZE}",
        f"--gradient_accumulation_steps={GRADIENT_ACCUMULATION}",
        f"--learning_rate={LEARNING_RATE}",
        f"--lr_scheduler={LR_SCHEDULER}",
        f"--lr_warmup_steps={LR_WARMUP_STEPS}",
        f"--max_train_steps={max_steps}",
        f"--rank={LORA_RANK}",
        f"--lora_dropout={LORA_DROPOUT}",
        "--use_dora",                    # ← THE critical DoRA flag
        f"--seed={SEED}",
        "--mixed_precision=fp16",
        "--checkpointing_steps=500",
        f"--validation_prompt={prompt}",
        "--validation_epochs=999",
        "--report_to=tensorboard",
        "--dataloader_num_workers=4",
    ]

    start = time.time()
    result = subprocess.run(cmd, text=True)
    elapsed_min = (time.time() - start) / 60

    if result.returncode == 0 and os.path.exists(weights_file):
        size_mb = os.path.getsize(weights_file) / 1024 / 1024
        print(f"  SUCCESS: {cls_name} DoRA saved ({size_mb:.1f} MB) in {elapsed_min:.1f} min")
        return True

    print(f"  FAILED: {cls_name} (exit {result.returncode}) after {elapsed_min:.1f} min")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune SDXL with DoRA on dermoscopic skin lesion images "
                    "(uses diffusers train_dreambooth_lora_sdxl.py --use_dora)"
    )
    parser.add_argument(
        "--train_data_dir", required=True,
        help="Root directory containing per-class subdirectories of training images"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Output directory for DoRA weights (one lora_{class}_final/ subdir per class)"
    )
    parser.add_argument(
        "--diffusers_dir", default=None,
        help="Optional path to a cloned huggingface/diffusers repository. "
             "If not provided, the script uses the bundled "
             "train_dreambooth_lora_sdxl.py (sourced from diffusers v0.37.1) "
             "that lives next to this wrapper."
    )
    parser.add_argument(
        "--classes", nargs="+", default=TARGET_CLASSES,
        help=f"Which classes to train (default: {' '.join(TARGET_CLASSES)})"
    )
    args = parser.parse_args()

    # resolve the training script once, up front — fail fast if missing
    train_script = resolve_train_script(args.diffusers_dir)
    if train_script is None:
        print("ERROR: could not find a DoRA training script.")
        print("  Either:")
        print("  1. Place train_dreambooth_lora_sdxl.py next to this wrapper, OR")
        print("  2. Pass --diffusers_dir pointing at a cloned diffusers repo")
        sys.exit(1)
    print(f"Using training script: {train_script}")

    os.makedirs(args.output_dir, exist_ok=True)

    print("SDXL DoRA TRAINING PLAN")
    print("=" * 60)
    print(f"Base model:      {SDXL_MODEL}")
    print(f"Training script: train_dreambooth_lora_sdxl.py --use_dora")
    print(f"DoRA rank:       {LORA_RANK}")
    print(f"Learning rate:   {LEARNING_RATE}  (NVIDIA: lower than LoRA)")
    print(f"LR scheduler:    {LR_SCHEDULER}")
    print(f"LoRA dropout:    {LORA_DROPOUT}")
    print(f"Resolution:      {RESOLUTION}x{RESOLUTION}")
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
        results[cls] = train_dora_for_class(
            cls, args.train_data_dir, args.output_dir, train_script
        )

    total_min = (time.time() - total_start) / 60
    print(f"\n{'=' * 60}")
    print(f"DORA TRAINING COMPLETE — {total_min:.1f} min total")
    print(f"{'=' * 60}")
    for cls, ok in results.items():
        print(f"  [{'OK ' if ok else 'FAIL'}] {cls}")

    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
