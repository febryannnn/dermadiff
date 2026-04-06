from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import PeftModel
import os
from PIL import Image
from tqdm import tqdm

SYNTHETIC_DIR = os.path.join(NOTEBOOK_DIR, 'temp/synthetic')

# Hitung jumlah training images per kelas minoritas dari splits
train_ids = set(SPLITS['train'])
DATA_ROOT = '/content/drive/MyDrive/DermaDiff/data/ham10000/images_classified'

minority_train_counts = {}
for cls_name in MINORITY_CLASSES:
    cls_path = os.path.join(DATA_ROOT, cls_name)
    if os.path.isdir(cls_path):
        count = sum(1 for f in os.listdir(cls_path)
                    if f.replace('.jpg','').replace('.png','') in train_ids)
        minority_train_counts[cls_name] = count

print("Minority class train counts:")
for cls, cnt in minority_train_counts.items():
    print(f"  {cls}: {cnt}")

# Generation ratios
RATIOS = {'1x': 1}

def generate_for_class(cls_name, num_images, lora_path, output_dir,
                       num_inference_steps=50, guidance_scale=9.0, batch_size=4):

    print(f"\n  Generating {num_images} images for {cls_name}...")

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
    pipe.to(DEVICE)

    os.makedirs(output_dir, exist_ok=True)

    prompts_pool = CLASS_PROMPTS[cls_name]
    negative_prompt = CLASS_NEGATIVE_PROMPTS[cls_name]

    generated = 0
    seed = 42

    with torch.no_grad():
        while generated < num_images:
            current_batch = min(batch_size, num_images - generated)
            generator = [torch.Generator(device=DEVICE).manual_seed(seed + generated + i)
                         for i in range(current_batch)]

            batch_prompts = [prompts_pool[(generated + i) % len(prompts_pool)]
                             for i in range(current_batch)]

            images = pipe(
                prompt=batch_prompts,
                negative_prompt=[negative_prompt] * current_batch,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images

            for i, img in enumerate(images):
                fname = f"sd20_synth_{cls_name}_{generated + i:05d}.png"
                img.save(os.path.join(output_dir, fname))

            generated += current_batch

            if generated % 20 == 0 or generated >= num_images:
                print(f"    {generated}/{num_images} done")

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return generated


# Generate untuk setiap ratio
# Generate untuk setiap ratio
for ratio_name, multiplier in RATIOS.items():
    print(f"\n{'='*60}")
    print(f"GENERATING {ratio_name} SYNTHETIC IMAGES")
    print(f"{'='*60}")

    ratio_dir = os.path.join(SYNTHETIC_DIR, ratio_name)

    for cls_name in MINORITY_CLASSES:
        output_dir = os.path.join(ratio_dir, cls_name)

        # ── SKIP kalau sudah ada images di folder ──
        if os.path.isdir(output_dir):
            existing = len([f for f in os.listdir(output_dir) if f.endswith(('.png', '.jpg'))])
            if existing > 0:
                print(f"{cls_name} ({ratio_name}): {existing} images sudah ada, SKIP!")
                continue

        lora_path = os.path.join(LORA_SAVE_DIR, f"lora_{cls_name}_best")
        if not os.path.exists(lora_path):
            lora_path = os.path.join(LORA_SAVE_DIR, f"lora_{cls_name}_final")

        if not os.path.exists(lora_path):
            print(f"No LoRA weights for {cls_name}, skipping!")
            continue

        num_to_generate = minority_train_counts.get(cls_name, 0) * multiplier
        if num_to_generate == 0:
            continue

        generate_for_class(cls_name, num_to_generate, lora_path, output_dir)

print("\nAll synthetic images generated!")

print("\nAll synthetic images generated!")

import matplotlib.pyplot as plt
from PIL import Image
import os
import random

# Count generated images
print("=" * 60)
print("SYNTHETIC IMAGE COUNTS")
print("=" * 60)

for ratio_name in ['5x']:
    ratio_dir = os.path.join(FILTERED_DIR_V2, ratio_name)
    if not os.path.exists(ratio_dir):
        print(f"  {ratio_name}: not generated yet")
        continue
    print(f"\n  {ratio_name}:")
    for cls_name in MINORITY_CLASSES:
        cls_dir = os.path.join(ratio_dir, cls_name)
        if os.path.isdir(cls_dir):
            count = len([f for f in os.listdir(cls_dir) if f.endswith(('.png', '.jpg'))])
            print(f"    {cls_name:10s}: {count}")

# Visualisasi sample (dari 1x ratio)
# Visualisasi sample — FILTERED 5x
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle("SD 2.0 + LoRA — Real vs Filtered Synthetic (5x)", fontsize=16)

for i, cls_name in enumerate(MINORITY_CLASSES):
    # Real sample
    real_paths = diffusion_data.get(cls_name, [])
    if real_paths:
        real_img = Image.open(random.choice(real_paths)).convert('RGB')
        axes[0, i].imshow(real_img)
        axes[0, i].set_title(f"REAL — {cls_name}", fontsize=12)
        axes[0, i].axis('off')

    # Filtered synthetic sample
    synth_dir = os.path.join(FILTERED_DIR_V2, '5x', cls_name)
    if os.path.isdir(synth_dir):
        synth_files = [f for f in os.listdir(synth_dir) if f.endswith(('.png', '.jpg'))]
        if synth_files:
            synth_img = Image.open(os.path.join(synth_dir, random.choice(synth_files))).convert('RGB')
            axes[1, i].imshow(synth_img)
            axes[1, i].set_title(f"FILTERED — {cls_name}", fontsize=12)
            axes[1, i].axis('off')

plt.tight_layout()
SAVE_DIR = os.path.join(NOTEBOOK_DIR, 'outputs')
plt.savefig(os.path.join(SAVE_DIR, 'synthetic_filtered_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Sample comparison saved!")