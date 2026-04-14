import os
import gc
import json
import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from torchvision import transforms

from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model

# Argument Parser 
def parse_args():
    parser = argparse.ArgumentParser(description="Train SD 2.1 + LoRA per class")

    # path config (default: relative to cwd)
    parser.add_argument("--project-root", type=str, default=".",
                        help="Project root folder (default: current directory)")
    parser.add_argument("--exp-name", type=str, default="exp_c2",
                        help="Experiment folder name under notebooks/")

    # model config
    parser.add_argument("--model-id", type=str,
                        default="Manojb/stable-diffusion-2-1-base",
                        help="HuggingFace model ID")
    parser.add_argument("--resolution", type=int, default=1024)

    # training config
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)

    # LoRA config
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    # classes to train
    parser.add_argument("--classes", nargs="+",
                        default=["mel", "bcc", "akiec", "df", "vasc"],
                        help="List of minority classes to generate")

    return parser.parse_args()

# Normalize Class Name 
def normalize_class(cls_name):
    cls_name = cls_name.lower()
    if "mel" in cls_name:
        return "mel"
    elif "bcc" in cls_name:
        return "bcc"
    elif "df" in cls_name:
        return "df"
    elif "vasc" in cls_name:
        return "vasc"
    elif "akiec" in cls_name or cls_name.startswith("ak"):
        return "akiec"
    else:
        return cls_name


# Class Prompts 
CLASS_PROMPTS = {
    "mel": [
        "a dermoscopic image of melanoma, irregular pigment network, blue-white veil, asymmetric pigmentation, multiple colors, irregular dots and globules, high magnification clinical dermoscopy",
        "a dermoscopic image of melanoma skin lesion, atypical network, regression structures, irregular streaks, blue-gray peppering, multicomponent pattern, sharp clinical dermoscopy image",
        "a dermoscopic image of malignant melanoma, asymmetric structure, irregular border, scar-like depigmentation, milky red areas, polymorphous vessels, clinical quality dermoscopy",
    ],
    "bcc": [
        "a dermoscopic image of basal cell carcinoma, arborizing vessels, blue-gray ovoid nests, maple leaf-like areas, spoke-wheel structures, shiny white lines, clinical quality dermoscopy",
        "a dermoscopic image of basal cell carcinoma, short fine telangiectasias, multiple small erosions, ulceration, blue-gray dots, crystalline structures, sharp focus dermoscopy",
        "a dermoscopic image of BCC skin lesion, tree-like branching vessels, white shiny areas, rosettes, milia-like cysts, well-defined border, high resolution clinical dermoscopy",
    ],
    "akiec": [
        "a dermoscopic image of actinic keratosis, strawberry pattern, rough scaly surface, rosette sign, white structureless areas, dotted vessels, clinical quality dermoscopy",
        "a dermoscopic image of actinic keratosis Bowen disease, red pseudo-network, surface scale, targetoid hair follicles, glomerular vessels, sharp focus clinical dermoscopy",
        "a dermoscopic image of intraepithelial carcinoma, keratinization, erythematous background, clustered dotted vessels, white halo around follicles, high magnification dermoscopy",
    ],
    "df": [
        "a dermoscopic image of dermatofibroma, central white scar-like patch, peripheral delicate pigment network, ring-like pattern, brown reticular lines at periphery, clinical quality dermoscopy",
        "a dermoscopic image of dermatofibroma skin lesion, white central area, faint peripheral pseudo-network, crystalline structures, shiny white lines, sharp focus dermoscopy",
        "a dermoscopic image of dermatofibroma, homogeneous white center, surrounding light brown pigment network, well-circumscribed border, smooth surface, clinical dermoscopy image",
    ],
    "vasc": [
        "a dermoscopic image of vascular lesion, red-purple lacunae, well-demarcated border, red-blue homogeneous areas, vascular spaces, clinical quality dermoscopy",
        "a dermoscopic image of vascular skin lesion, dark red to purple lagoons, whitish veil, thrombosed areas, sharply defined margin, high resolution dermoscopy",
        "a dermoscopic image of hemangioma, red-blue globular structures, lacunar pattern, well-circumscribed border, smooth surface, clinical quality dermoscopy image",
    ],
}

# negative prompts
CLASS_NEGATIVE_PROMPTS = {
    "mel": "benign nevus, symmetric, uniform color, regular border, blurry, low quality, artifacts, text, watermark, cartoon, non-dermoscopic, overexposed",
    "bcc": "melanoma, nevus, pigment network, blurry, low quality, artifacts, text, watermark, cartoon, non-dermoscopic, overexposed",
    "akiec": "melanoma, smooth surface, no scale, blurry, low quality, artifacts, text, watermark, cartoon, non-dermoscopic, overexposed",
    "df": "melanoma, irregular border, blue-white veil, blurry, low quality, artifacts, text, watermark, cartoon, non-dermoscopic, overexposed",
    "vasc": "melanoma, pigment network, brown color, blurry, low quality, artifacts, text, watermark, cartoon, non-dermoscopic, overexposed",
}


# Dataset Class
class DermDiffusionDataset(Dataset):
    # initialize dataset with image paths and class name
    def __init__(self, image_paths, class_name, resolution=1024):
        self.image_paths = image_paths
        self.class_name = class_name
        self.resolution = resolution
        # image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)
        # rotate through available prompts
        prompts_pool = CLASS_PROMPTS.get(
            self.class_name,
            [f"a dermoscopic image of {self.class_name} skin lesion, clinical quality, high resolution"]
        )
        prompt = prompts_pool[idx % len(prompts_pool)]
        return {"pixel_values": img, "prompt": prompt}

# Data Collection
def collect_diffusion_data(data_sources):
    diffusion_data = defaultdict(list)

    for source_name, source_dir in data_sources.items():
        if not os.path.exists(source_dir):
            print(f"{source_name} not found: {source_dir}")
            continue
        for cls_folder in sorted(os.listdir(source_dir)):
            cls_path = os.path.join(source_dir, cls_folder)
            if not os.path.isdir(cls_path):
                continue
            # normalize class name (lowercase)
            cls_name = normalize_class(cls_folder)
            for img_file in os.listdir(cls_path):
                if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    diffusion_data[cls_name].append(os.path.join(cls_path, img_file))

    return diffusion_data


def print_data_summary(diffusion_data, minority_classes):
    print("\nTraining Data Summary")
    print("=" * 50)

    minority_total = 0
    non_minority_total = 0

    # get all available classes
    all_classes = sorted(diffusion_data.keys())

    print("\nPer Class Count:")
    print("-" * 50)

    for cls in all_classes:
        count = len(diffusion_data.get(cls, []))
        if cls in minority_classes:
            label = "MINORITY"
            minority_total += count
        else:
            label = "NON-MINORITY"
            non_minority_total += count
        print(f"{cls:10s}: {count:6d}  ({label})")

    print("-" * 50)
    print(f"{'TOTAL MINORITY':15s}: {minority_total:6d}")
    print(f"{'TOTAL NON-MINORITY':15s}: {non_minority_total:6d}")
    print(f"{'GRAND TOTAL':15s}: {minority_total + non_minority_total:6d}")


# Preview Sample
def save_sample_preview(diffusion_data, cls, output_path):
    sample_paths = diffusion_data[cls][:6]
    if len(sample_paths) == 0:
        print(f"No samples for {cls}, skipping preview")
        return

    dataset = DermDiffusionDataset(sample_paths, class_name=cls)
    samples = [dataset[i] for i in range(len(sample_paths))]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, ax in enumerate(axes.flatten()):
        img = samples[i]["pixel_values"]
        prompt = samples[i]["prompt"]
        img = (img + 1) / 2
        img = img.permute(1, 2, 0).cpu().numpy()
        ax.imshow(img)
        ax.set_title(prompt[:60] + "...", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Sample preview saved: {output_path}")

# Load Pretrained Components
def load_sd_components(model_id, device):
    print("Loading tokenizer & text encoder...")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=torch.float16
    )

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float16
    )

    print("Loading scheduler...")
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # freeze VAE and text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    vae.to(device)
    text_encoder.to(device)

    return tokenizer, text_encoder, vae, noise_scheduler


def build_lora_unet(model_id, lora_config, device):
    unet_base = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=torch.float16
    )
    unet = get_peft_model(unet_base, lora_config)
    unet.to(device)
    return unet

# Training Loop
def train_lora_for_class(
    class_name,
    image_paths,
    unet,
    vae,
    text_encoder,
    tokenizer,
    noise_scheduler,
    device,
    num_epochs=100,
    batch_size=4,
    learning_rate=1e-4,
    gradient_accumulation_steps=4,
    num_workers=2,
    resolution=1024,
    save_dir=None,
):
    print(f"Training LoRA for class: {class_name.upper()}")
    print(f"  Images: {len(image_paths)}, Epochs: {num_epochs}, BS: {batch_size}")
    print(f"  Effective BS: {batch_size * gradient_accumulation_steps}")
    print(f"{'='*60}")

    dataset = DermDiffusionDataset(image_paths, class_name, resolution=resolution)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    # optimizer only for LoRA parameters
    optimizer = AdamW(
        [p for p in unet.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=1e-2,
    )

    # LR scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs * len(dataloader)
    )

    scaler = GradScaler(enabled=(device == "cuda"))
    unet.train()
    global_step = 0
    best_loss = float("inf")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(dataloader):
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
            prompts = batch["prompt"]

            # tokenize prompts
            text_inputs = tokenizer(
                prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(device)

            with autocast(device_type=device, enabled=(device == "cuda")):
                # encode text
                encoder_hidden_states = text_encoder(text_input_ids)[0]

                # encode images to latent space
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # sample noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=device
                ).long()

                # add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # predict noise
                noise_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample

                # MSE loss
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                loss = loss / gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                lr_scheduler.step()
                global_step += 1

            epoch_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f"  Epoch {epoch+1:03d}/{num_epochs} | Loss: {avg_loss:.6f} | LR: {lr_scheduler.get_last_lr()[0]:.2e}")

        # save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            if save_dir:
                best_path = os.path.join(save_dir, f"lora_{class_name}_best")
                unet.save_pretrained(best_path)

    # save final
    if save_dir:
        final_path = os.path.join(save_dir, f"lora_{class_name}_final")
        unet.save_pretrained(final_path)
        print(f"Saved to {final_path}")

    print(f"Best loss: {best_loss:.6f}")
    return best_loss


# Main
def main():
    args = parse_args()

    # disable wandb
    os.environ["WANDB_MODE"] = "disabled"

    # resolve project paths
    project_root = os.path.abspath(args.project_root)
    shared_dir = os.path.join(project_root, "shared")
    notebook_dir = os.path.join(project_root, "notebooks", args.exp_name)

    # output directory for test and val predictions
    os.makedirs(os.path.join(notebook_dir, "outputs"), exist_ok=True)
    # temp directory for LoRA weights and synthetic images
    os.makedirs(os.path.join(notebook_dir, "temp"), exist_ok=True)

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch: {torch.__version__}")
    print(f"Device : {device}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    else:
        print("GPU    : No GPU detected (training will be very slow on CPU)")

    # shared config across all stable diffusion experiments
    with open(os.path.join(shared_dir, "config/shared_config.json")) as f:
        CONFIG = json.load(f)
    with open(os.path.join(shared_dir, "config/label_mapping.json")) as f:
        LABEL_MAP = json.load(f)
    with open(os.path.join(shared_dir, "splits/ham10000_splits.json")) as f:
        SPLITS = json.load(f)

    print(f"\nConfig: {CONFIG['model_name']}, {CONFIG['nb_classes']} classes")
    print(f"Split: train={len(SPLITS['train'])}, val={len(SPLITS['val'])}, test={len(SPLITS['test'])}")
    print(f"Label map: {LABEL_MAP}")

    # diffusion model datasets (ham10000, isic2019, and longitudinal)
    data_sources = {
        "ham10000": os.path.join(project_root, "data/ham10000/images_classified"),
        "isic2019": os.path.join(project_root, "data/isic2019/images"),
        "longitudinal": os.path.join(shared_dir, "diffusion_extra"),
    }

    # collect all image paths per class (for diffusion training)
    minority_classes = args.classes
    diffusion_data = collect_diffusion_data(data_sources)
    print_data_summary(diffusion_data, minority_classes)

    # preview sample (saved to file, no window display)
    preview_cls = minority_classes[0] if minority_classes else "mel"
    if preview_cls in diffusion_data and len(diffusion_data[preview_cls]) > 0:
        preview_path = os.path.join(notebook_dir, "outputs", f"sample_{preview_cls}.png")
        save_sample_preview(diffusion_data, preview_cls, preview_path)

    # set env var HF_TOKEN before running this script
    if not os.environ.get("HF_TOKEN") and not os.path.exists(
        os.path.expanduser("~/.cache/huggingface/token")
    ):
        print("\nWarning: HF token not found.")
        print("Run `huggingface-cli login` or `export HF_TOKEN=...` first\n")

    # load SD components (shared across all classes)
    tokenizer, text_encoder, vae, noise_scheduler = load_sd_components(
        args.model_id, device
    )

    # LoRA config for UNet
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=[
            "to_k", "to_q", "to_v", "to_out.0",
            "proj_in", "proj_out",
            "ff.net.0.proj", "ff.net.2",
        ],
        lora_dropout=args.lora_dropout,
    )

    print(f"\nStable Diffusion loaded on {device}!")

    # LoRA save dir
    lora_save_dir = os.path.join(notebook_dir, "temp/lora_weights")
    os.makedirs(lora_save_dir, exist_ok=True)

    # train per class
    training_results = {}

    for cls_name in minority_classes:
        # skip if already trained
        final_path = os.path.join(lora_save_dir, f"lora_{cls_name}_final")
        if os.path.exists(final_path):
            print(f"\n{cls_name} already exists, SKIP!")
            continue

        if cls_name not in diffusion_data or len(diffusion_data[cls_name]) == 0:
            print(f"\nNo data for {cls_name}, skipping...")
            continue

        # build fresh LoRA UNet per class
        unet = build_lora_unet(args.model_id, lora_config, device)
        unet.print_trainable_parameters()

        best_loss = train_lora_for_class(
            class_name=cls_name,
            image_paths=diffusion_data[cls_name],
            unet=unet,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            noise_scheduler=noise_scheduler,
            device=device,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_workers=args.num_workers,
            resolution=args.resolution,
            save_dir=lora_save_dir,
        )

        training_results[cls_name] = best_loss

        del unet
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # summary
    print("\n" + "=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)
    for cls, loss in training_results.items():
        print(f"  {cls:10s}: best_loss = {loss:.6f}")


if __name__ == "__main__":
    main()