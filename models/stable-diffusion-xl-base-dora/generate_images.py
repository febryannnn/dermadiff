#!/usr/bin/env python3
"""
DermaDiff — Phase 2 (DoRA): Synthetic Image Generation
========================================================

Generates synthetic dermoscopic skin lesion images using SDXL with the DoRA
adapters trained in Phase 1 (DoRA variant).

Identical to the LoRA generation script in every respect EXCEPT:
  - Generated filenames are prefixed with `dora_sdxl_` instead of `sdxl_lora_`
  - The default --lora_dir points at this folder's own "LoRA Weights" subdir

The diffusers `pipe.load_lora_weights()` call auto-detects whether the
safetensors file contains DoRA (magnitude vectors) or plain LoRA, so no
code changes are needed at the loading level. Produces a configurable multiple of each class's
real training count (default 2x), with crash-safe incremental saving.

Usage
-----
    python generate_images.py \\
        --lora_dir /path/to/dora_weights \\
        --output_dir /path/to/synthetic_output \\
        --train_counts mel=779 bcc=360 akiec=229 df=81 vasc=99
"""

import argparse
import math
import os
import sys
import time

import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────────────

SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.5
RESOLUTION = 1024
MAX_RATIO = 2          # Generate 2x; subsample to 1x downstream
BATCH_SIZE = 1         # 1024x1024 needs batch=1 to avoid OOM on most GPUs
BASE_SEED = 42

TARGET_CLASSES = ["mel", "bcc", "akiec", "df", "vasc"]

# Five rotating prompt variants per class for diverse generation.
# These detailed dermoscopic prompts encourage the model to produce varied
# clinically relevant patterns rather than collapsing to a single look.
CLASS_PROMPTS = {
    "mel": [
        "a dermoscopic photograph of melanoma with irregular pigment network, multiple colors including brown black and blue-gray, asymmetric borders, on light skin",
        "a dermoscopic close-up of melanoma skin lesion showing atypical pigment network with blue-white veil, irregular dots and globules, on tan skin background",
        "a clinical dermoscopic image of melanoma with irregular streaks, regression structures, multiple shades of brown and dark pigmentation, scar-like depigmentation",
        "a dermoscopic photograph of superficial spreading melanoma with asymmetric pigmentation, dark brown to black blotches, irregular border, on fair skin",
        "a dermoscopic image of early melanoma showing atypical network with wide irregular meshes, brown and blue-gray colors, structureless areas",
    ],
    "bcc": [
        "a dermoscopic photograph of basal cell carcinoma with arborizing telangiectasia vessels, blue-gray ovoid nests, translucent pearly background on light skin",
        "a dermoscopic close-up of pigmented basal cell carcinoma showing leaf-like areas, blue-gray globules, branching tree-like vessels, ulceration",
        "a clinical dermoscopic image of nodular basal cell carcinoma with bright red arborizing vessels, blue-gray structures, shiny white areas on pale skin",
        "a dermoscopic photograph of basal cell carcinoma with spoke-wheel structures, multiple blue-gray dots, fine telangiectasia, on skin colored background",
        "a dermoscopic image of BCC showing large blue-gray ovoid nests, arborizing vessels with fine branches, absence of pigment network",
    ],
    "akiec": [
        "a dermoscopic photograph of actinic keratosis with strawberry pattern, red pseudonetwork, white-yellow follicular plugs, surface scale on sun-damaged skin",
        "a dermoscopic close-up of actinic keratosis showing erythematous background, prominent yellowish hair follicles with white halo, fine wavy vessels",
        "a clinical dermoscopic image of actinic keratosis with white opaque scales on reddish background, rosettes, targetoid follicular openings on aged skin",
        "a dermoscopic photograph of keratosis showing rough scaly surface, red erythema, prominent follicular openings surrounded by white circles, on fair skin",
        "a dermoscopic image of actinic keratosis with fine linear vessels, white-yellow keratotic surface, erythematous pseudonetwork pattern",
    ],
    "df": [
        "a dermoscopic photograph of dermatofibroma with central white scar-like patch surrounded by delicate peripheral brown pigment network on skin",
        "a dermoscopic close-up of dermatofibroma showing central white fibrotic area with peripheral light brown reticular network that fades into surrounding skin",
        "a clinical dermoscopic image of dermatofibroma with bright white central patch, dotted vessels, delicate tan peripheral network on light-to-medium skin tone",
        "a dermoscopic photograph of dermatofibroma with homogeneous white center and ring-like brown structures at periphery, smooth dome-shaped on leg skin",
        "a dermoscopic image of dermatofibroma showing crystalline white structures in center with surrounding fine pigment network, brown to tan coloration",
    ],
    "vasc": [
        "a dermoscopic photograph of vascular lesion with well-demarcated red and blue-red lacunae of varying sizes clustered against reddish background",
        "a dermoscopic close-up of cherry angioma showing multiple round red lacunae separated by whitish septa on skin surface",
        "a clinical dermoscopic image of vascular skin lesion with dark red to purple lacunae, reddish-blue homogeneous background, dilated blood vessels",
        "a dermoscopic photograph of angioma with clustered red-blue lacunae varying in size, thrombosed dark areas, against red homogeneous background",
        "a dermoscopic image of vascular lesion showing well-defined round to oval red lacunae with reddish-brown coloration and whitish fibrous septa",
    ],
}


def generate_for_class(
    pipe: StableDiffusionXLPipeline,
    cls_name: str,
    num_images: int,
    output_dir: str,
    lora_dir: str,
) -> int:
    """Generate `num_images` synthetic images for one class. Crash-safe."""

    cls_out = os.path.join(output_dir, cls_name)
    os.makedirs(cls_out, exist_ok=True)

    # Resume support: skip if already complete
    existing = len([f for f in os.listdir(cls_out) if f.endswith(".jpg")])
    if existing >= num_images:
        print(f"  SKIP {cls_name}: {existing} images already present")
        return existing

    lora_path = os.path.join(lora_dir, f"lora_{cls_name}_final")
    pipe.load_lora_weights(lora_path)

    prompts = CLASS_PROMPTS[cls_name]
    generated = 0
    num_batches = math.ceil(num_images / BATCH_SIZE)

    print(f"  Generating {num_images} images for {cls_name} ({num_batches} batches)...")

    for batch_idx in tqdm(range(num_batches), desc=f"  {cls_name}"):
        batch_count = min(BATCH_SIZE, num_images - generated)
        batch_prompts = [
            prompts[(batch_idx * BATCH_SIZE + i) % len(prompts)]
            for i in range(batch_count)
        ]

        seed = BASE_SEED + batch_idx * BATCH_SIZE
        generator = torch.Generator(device="cuda").manual_seed(seed)

        with torch.no_grad():
            result = pipe(
                prompt=batch_prompts,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                height=RESOLUTION,
                width=RESOLUTION,
                generator=generator,
            )

        for i, img in enumerate(result.images):
            img_idx = generated + i
            fname = f"dora_sdxl_{cls_name}_{img_idx:04d}.jpg"
            img.save(os.path.join(cls_out, fname), quality=95)

        generated += batch_count

        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()

    pipe.unload_lora_weights()
    torch.cuda.empty_cache()
    return generated


def parse_train_counts(items):
    """Parse 'class=count' strings into a dict."""
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--train_counts must be CLASS=N pairs, got: {item}")
        k, v = item.split("=", 1)
        out[k.strip()] = int(v)
    return out


def derive_train_counts(splits_json: str, ham_metadata: str, classes: list) -> dict:
    """Derive per-class HAM10000 train-split counts from the splits JSON.

    Reads the train image IDs from splits_json, looks up each in the HAM
    metadata CSV, and counts per-class. This is the auto-mode alternative
    to passing --train_counts manually.
    """
    import json
    import pandas as pd

    with open(splits_json) as f:
        splits = json.load(f)
    train_ids = set(splits["train"])
    id_col = splits["metadata"]["id_column"]

    ham_df = pd.read_csv(ham_metadata)
    ham_train = ham_df[ham_df[id_col].isin(train_ids)]

    counts = {cls: int((ham_train["dx"] == cls).sum()) for cls in classes}
    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic dermoscopic images using SDXL + DoRA"
    )
    # Default --lora_dir to the in-repo "LoRA Weights" folder that lives
    # next to this script. Computed from __file__ so it works regardless
    # of which directory the user runs the script from.
    DEFAULT_LORA_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "LoRA Weights"
    )

    parser.add_argument(
        "--lora_dir", default=DEFAULT_LORA_DIR,
        help=f"Directory containing per-class LoRA weight subdirectories "
             f"(default: in-repo LoRA Weights folder at {DEFAULT_LORA_DIR})"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Output directory for generated synthetic images (per-class subdirs)"
    )
    parser.add_argument(
        "--train_counts", nargs="+", default=None,
        help="Per-class real training counts as CLASS=N pairs (e.g. mel=779 bcc=360). "
             "If omitted, derive from --splits_json + --ham_metadata."
    )
    parser.add_argument(
        "--splits_json", default=None,
        help="HAM10000 splits JSON from Phase 0 (alternative to --train_counts)"
    )
    parser.add_argument(
        "--ham_metadata", default=None,
        help="HAM10000_metadata.csv (required with --splits_json)"
    )
    parser.add_argument(
        "--ratio", type=int, default=MAX_RATIO,
        help=f"Synthetic-to-real ratio (default: {MAX_RATIO})"
    )
    parser.add_argument(
        "--classes", nargs="+", default=TARGET_CLASSES,
        help=f"Which classes to generate (default: {' '.join(TARGET_CLASSES)})"
    )
    args = parser.parse_args()

    if args.train_counts:
        train_counts = parse_train_counts(args.train_counts)
        print(f"Using manual --train_counts: {train_counts}")
    elif args.splits_json and args.ham_metadata:
        train_counts = derive_train_counts(
            args.splits_json, args.ham_metadata, args.classes
        )
        print(f"Derived train counts from splits JSON: {train_counts}")
    else:
        print("ERROR: must provide either --train_counts OR "
              "(--splits_json AND --ham_metadata)")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Verify all LoRA weights exist before loading the pipeline
    print("Checking LoRA weights...")
    missing = []
    for cls in args.classes:
        wf = os.path.join(args.lora_dir, f"lora_{cls}_final", "pytorch_lora_weights.safetensors")
        if not os.path.exists(wf):
            missing.append(cls)
            print(f"  [MISSING] {cls}")
        else:
            size_mb = os.path.getsize(wf) / 1024 / 1024
            print(f"  [OK]      {cls}: {size_mb:.1f} MB")
    if missing:
        print(f"\nERROR: missing LoRA weights for: {missing}")
        sys.exit(1)

    print(f"\nLoading SDXL base model: {SDXL_MODEL}")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL_MODEL,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to("cuda")
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    print("SDXL pipeline ready.")

    print("\nGENERATION PLAN")
    print("=" * 60)
    print(f"Resolution: {RESOLUTION}x{RESOLUTION}")
    print(f"Steps:      {NUM_INFERENCE_STEPS}")
    print(f"Guidance:   {GUIDANCE_SCALE}")
    print(f"Ratio:      {args.ratio}x")
    print()

    total_target = 0
    for cls in args.classes:
        n_target = train_counts[cls] * args.ratio
        total_target += n_target
        print(f"  {cls:6s}: {train_counts[cls]:4d} train x {args.ratio} = {n_target:5d}")
    print(f"\nTotal target: {total_target} images")

    results = {}
    total_start = time.time()
    for cls in args.classes:
        n_target = train_counts[cls] * args.ratio
        start = time.time()
        print(f"\n{'=' * 60}")
        print(f"  {cls}: generating {n_target} images")
        print(f"{'=' * 60}")
        count = generate_for_class(pipe, cls, n_target, args.output_dir, args.lora_dir)
        elapsed_min = (time.time() - start) / 60
        results[cls] = count
        print(f"  Done: {count} images in {elapsed_min:.1f} min")

    total_min = (time.time() - total_start) / 60
    print(f"\n{'=' * 60}")
    print(f"GENERATION COMPLETE — {total_min:.1f} min total")
    print(f"{'=' * 60}")
    for cls, count in results.items():
        target = train_counts[cls] * args.ratio
        status = "OK" if count >= target else "LOW"
        print(f"  [{status}] {cls}: {count}/{target}")


if __name__ == "__main__":
    main()
