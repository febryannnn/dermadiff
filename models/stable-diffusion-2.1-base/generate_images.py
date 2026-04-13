import os
import gc
import json
import random
import argparse

import torch
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import PeftModel

# reuse prompts from training script
from train_sd21_lora import (
    CLASS_PROMPTS,
    CLASS_NEGATIVE_PROMPTS,
    collect_diffusion_data,
)


# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic images using SD 2.1 + LoRA")

    # path config
    parser.add_argument("--project-root", type=str, default=".",
                        help="Project root folder (default: current directory)")
    parser.add_argument("--exp-name", type=str, default="exp_c2",
                        help="Experiment folder name under notebooks/")

    # model
    parser.add_argument("--model-id", type=str,
                        default="Manojb/stable-diffusion-2-1-base")

    # classes
    parser.add_argument("--classes", nargs="+",
                        default=["mel", "bcc", "akiec", "df", "vasc"])

    # generation ratios
    parser.add_argument("--ratios", nargs="+", default=["1x"],
                        help="Generation ratios, e.g. 1x 5x")

    # inference config
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=9.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # visualization
    parser.add_argument("--viz-ratio", type=str, default=None,
                        help="Ratio to use for real-vs-synthetic visualization (default: last ratio)")

    return parser.parse_args()


# Parse Ratio String
def parse_ratio(ratio_str):
    # "1x" -> 1, "5x" -> 5
    return int(ratio_str.rstrip("x"))


# Count Minority Training Images
def count_minority_train_images(data_root, minority_classes, train_ids):
    minority_train_counts = {}
    for cls_name in minority_classes:
        cls_path = os.path.join(data_root, cls_name)
        if os.path.isdir(cls_path):
            count = sum(
                1 for f in os.listdir(cls_path)
                if f.replace(".jpg", "").replace(".png", "") in train_ids
            )
            minority_train_counts[cls_name] = count
    return minority_train_counts


# Generate For Class
def generate_for_class(
    cls_name,
    num_images,
    lora_path,
    output_dir,
    model_id,
    device,
    num_inference_steps=50,
    guidance_scale=9.0,
    batch_size=4,
    seed=42,
):
    print(f"\n  Generating {num_images} images for {cls_name}...")

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
    pipe.to(device)

    os.makedirs(output_dir, exist_ok=True)

    prompts_pool = CLASS_PROMPTS[cls_name]
    negative_prompt = CLASS_NEGATIVE_PROMPTS[cls_name]

    generated = 0

    with torch.no_grad():
        while generated < num_images:
            current_batch = min(batch_size, num_images - generated)
            generator = [
                torch.Generator(device=device).manual_seed(seed + generated + i)
                for i in range(current_batch)
            ]

            batch_prompts = [
                prompts_pool[(generated + i) % len(prompts_pool)]
                for i in range(current_batch)
            ]

            images = pipe(
                prompt=batch_prompts,
                negative_prompt=[negative_prompt] * current_batch,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images

            for i, img in enumerate(images):
                fname = f"sd21_synth_{cls_name}_{generated + i:05d}.png"
                img.save(os.path.join(output_dir, fname))

            generated += current_batch

            if generated % 20 == 0 or generated >= num_images:
                print(f"    {generated}/{num_images} done")

    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return generated


# Save Real vs Synthetic Comparison
def save_comparison_plot(
    diffusion_data, synthetic_dir, ratio_name, minority_classes, output_path
):
    fig, axes = plt.subplots(2, len(minority_classes), figsize=(4 * len(minority_classes), 8))
    fig.suptitle(f"SD 2.1 + LoRA - Real vs Synthetic ({ratio_name})", fontsize=16)

    # handle case where only 1 class (axes is 1D)
    if len(minority_classes) == 1:
        axes = axes.reshape(2, 1)

    for i, cls_name in enumerate(minority_classes):
        # real sample
        real_paths = diffusion_data.get(cls_name, [])
        if real_paths:
            real_img = Image.open(random.choice(real_paths)).convert("RGB")
            axes[0, i].imshow(real_img)
            axes[0, i].set_title(f"REAL - {cls_name}", fontsize=12)
            axes[0, i].axis("off")

        # synthetic sample
        synth_dir = os.path.join(synthetic_dir, ratio_name, cls_name)
        if os.path.isdir(synth_dir):
            synth_files = [f for f in os.listdir(synth_dir) if f.endswith((".png", ".jpg"))]
            if synth_files:
                synth_img = Image.open(os.path.join(synth_dir, random.choice(synth_files))).convert("RGB")
                axes[1, i].imshow(synth_img)
                axes[1, i].set_title(f"SYNTH - {cls_name}", fontsize=12)
                axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved: {output_path}")


# Main
def main():
    args = parse_args()

    # resolve project paths
    project_root = os.path.abspath(args.project_root)
    shared_dir = os.path.join(project_root, "shared")
    notebook_dir = os.path.join(project_root, "notebooks", args.exp_name)

    synthetic_dir = os.path.join(notebook_dir, "temp/synthetic")
    lora_save_dir = os.path.join(notebook_dir, "temp/lora_weights")
    output_viz_dir = os.path.join(notebook_dir, "outputs")
    os.makedirs(output_viz_dir, exist_ok=True)

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if not torch.cuda.is_available():
        print("Warning: No GPU detected (generation will be very slow on CPU)")

    # load splits to get training image IDs
    with open(os.path.join(shared_dir, "splits/ham10000_splits.json")) as f:
        SPLITS = json.load(f)

    train_ids = set(SPLITS["train"])

    # count training images per minority class
    data_root = os.path.join(project_root, "data/ham10000/images_classified")
    minority_classes = args.classes
    minority_train_counts = count_minority_train_images(
        data_root, minority_classes, train_ids
    )

    print("\nMinority class train counts:")
    for cls, cnt in minority_train_counts.items():
        print(f"  {cls}: {cnt}")

    # build ratios dict from args
    ratios = {r: parse_ratio(r) for r in args.ratios}

    # generate per ratio
    for ratio_name, multiplier in ratios.items():
        print(f"\n{'='*60}")
        print(f"GENERATING {ratio_name} SYNTHETIC IMAGES")
        print(f"{'='*60}")

        ratio_dir = os.path.join(synthetic_dir, ratio_name)

        for cls_name in minority_classes:
            output_dir = os.path.join(ratio_dir, cls_name)

            # skip if images already exist in folder
            if os.path.isdir(output_dir):
                existing = len([f for f in os.listdir(output_dir) if f.endswith((".png", ".jpg"))])
                if existing > 0:
                    print(f"{cls_name} ({ratio_name}): {existing} images already exist, SKIP!")
                    continue

            # prefer best checkpoint, fallback to final
            lora_path = os.path.join(lora_save_dir, f"lora_{cls_name}_best")
            if not os.path.exists(lora_path):
                lora_path = os.path.join(lora_save_dir, f"lora_{cls_name}_final")

            if not os.path.exists(lora_path):
                print(f"No LoRA weights for {cls_name}, skipping!")
                continue

            num_to_generate = minority_train_counts.get(cls_name, 0) * multiplier
            if num_to_generate == 0:
                continue

            generate_for_class(
                cls_name=cls_name,
                num_images=num_to_generate,
                lora_path=lora_path,
                output_dir=output_dir,
                model_id=args.model_id,
                device=device,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                batch_size=args.batch_size,
                seed=args.seed,
            )

    print("\nAll synthetic images generated!")

    # print counts summary
    print("\n" + "=" * 60)
    print("SYNTHETIC IMAGE COUNTS")
    print("=" * 60)
    for ratio_name in ratios.keys():
        ratio_dir = os.path.join(synthetic_dir, ratio_name)
        if not os.path.exists(ratio_dir):
            print(f"  {ratio_name}: not generated yet")
            continue
        print(f"\n  {ratio_name}:")
        for cls_name in minority_classes:
            cls_dir = os.path.join(ratio_dir, cls_name)
            if os.path.isdir(cls_dir):
                count = len([f for f in os.listdir(cls_dir) if f.endswith((".png", ".jpg"))])
                print(f"    {cls_name:10s}: {count}")

    # real vs synthetic comparison visualization
    viz_ratio = args.viz_ratio or list(ratios.keys())[-1]
    data_sources = {
        "ham10000": os.path.join(project_root, "data/ham10000/images_classified"),
        "isic2019": os.path.join(project_root, "data/isic2019/images"),
        "longitudinal": os.path.join(shared_dir, "diffusion_extra"),
    }
    diffusion_data = collect_diffusion_data(data_sources)

    comparison_path = os.path.join(output_viz_dir, f"synthetic_comparison_{viz_ratio}.png")
    save_comparison_plot(
        diffusion_data, synthetic_dir, viz_ratio, minority_classes, comparison_path
    )


if __name__ == "__main__":
    main()