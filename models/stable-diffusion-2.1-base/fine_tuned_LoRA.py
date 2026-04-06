import subprocess, sys, os

# Install dependencies
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "diffusers==0.30.3", "transformers==4.44.2", "accelerate==0.34.2",
    "peft", "bitsandbytes", "datasets", "pillow", "tqdm"
])

# restart session jika versi transformers tidak sesuai
import transformers
if transformers.__version__ != "4.44.2":
    print("Restarting runtime...")
    os.kill(os.getpid(), 9)

import torch
print(f"transformers={transformers.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")

import torch
print("PyTorch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Tidak ada GPU!")
print("CUDA:", torch.cuda.is_available())

from google.colab import drive
drive.mount('/content/drive')

import os, json
os.environ['WANDB_MODE'] = 'disabled'

# setup path project (folder DermaDiff)
PROJECT_ROOT = '/content/drive/MyDrive/DermaDiff'
SHARED = os.path.join(PROJECT_ROOT, 'shared')

# setup path ke notebook
NOTEBOOK_DIR = os.path.join(PROJECT_ROOT, 'notebooks/exp_c2')

# direktori output untuk menyimpan hasil test dan val prediction
os.makedirs(os.path.join(NOTEBOOK_DIR, 'outputs'), exist_ok=True)

# direktori temp untuk menyimpan lora weight dan hasil sintetis gambar
os.makedirs(os.path.join(NOTEBOOK_DIR, 'temp'), exist_ok=True)

# setup config agar sama untuk notebook stable diffusion yang lain
with open(os.path.join(SHARED, 'config/shared_config.json')) as f:
    CONFIG = json.load(f)

# setup mapping label class agar sama untuk notebook stable diffusion yang lain
with open(os.path.join(SHARED, 'config/label_mapping.json')) as f:
    LABEL_MAP = json.load(f)

# setup mapping split data (train, val, test) class agar sama untuk notebook stable diffusion yang lain
with open(os.path.join(SHARED, 'splits/ham10000_splits.json')) as f:
    SPLITS = json.load(f)

# Load path weight model untuk PanDerm Classifier
WEIGHT_FILE = os.path.join(SHARED, 'weights/panderm_ll_data6_checkpoint-499.pth')

print(f"Config: {CONFIG['model_name']}, {CONFIG['nb_classes']} classes")
print(f"Split: train={len(SPLITS['train'])}, val={len(SPLITS['val'])}, test={len(SPLITS['test'])}")
print(f"Label map: {LABEL_MAP}")

import os
import json
from pathlib import Path
from collections import defaultdict

# Dataset diffusion model (ham10000, isic2019, dan longitudinal)
DATA_SOURCES = {
    'ham10000': '/content/drive/MyDrive/DermaDiff/data/ham10000/images_classified',
    'isic2019': '/content/drive/MyDrive/DermaDiff/data/isic2019/images',
    'longitudinal': '/content/drive/MyDrive/DermaDiff/shared/diffusion_extra',
}

# define 5 kelas minoritas yang perlu di generate
MINORITY_CLASSES = ['mel', 'bcc', 'akiec', 'df', 'vasc']

def normalize_class(cls_name):
    cls_name = cls_name.lower()

    if 'mel' in cls_name:
        return 'mel'
    elif 'bcc' in cls_name:
        return 'bcc'
    elif 'df' in cls_name:
        return 'df'
    elif 'vasc' in cls_name:
        return 'vasc'
    elif 'akiec' in cls_name or cls_name.startswith('ak'):
        return 'akiec'
    else:
        return cls_name

# Kumpulkan semua image paths per kelas (untuk training diffusion)
diffusion_data = defaultdict(list)

for source_name, source_dir in DATA_SOURCES.items():
    if not os.path.exists(source_dir):
        print(f"{source_name} not found: {source_dir}")
        continue
    for cls_folder in sorted(os.listdir(source_dir)):
        cls_path = os.path.join(source_dir, cls_folder)
        if not os.path.isdir(cls_path):
            continue
        # Normalize nama kelas (lowercase)
        cls_name = normalize_class(cls_folder)
        for img_file in os.listdir(cls_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                diffusion_data[cls_name].append(os.path.join(cls_path, img_file))

print("\nTraining Data Summary")
print("=" * 50)

minority_total = 0
non_minority_total = 0

# Ambil semua kelas yang ada
all_classes = sorted(diffusion_data.keys())

print("\nPer Class Count:")
print("-" * 50)

for cls in all_classes:
    count = len(diffusion_data.get(cls, []))

    if cls in MINORITY_CLASSES:
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

# Prompt per kelas minoritas

CLASS_PROMPTS = {
    'mel': [
        "a dermoscopic image of melanoma, irregular pigment network, blue-white veil, asymmetric pigmentation, multiple colors, irregular dots and globules, high magnification clinical dermoscopy",
        "a dermoscopic image of melanoma skin lesion, atypical network, regression structures, irregular streaks, blue-gray peppering, multicomponent pattern, sharp clinical dermoscopy image",
        "a dermoscopic image of malignant melanoma, asymmetric structure, irregular border, scar-like depigmentation, milky red areas, polymorphous vessels, clinical quality dermoscopy",
    ],
    'bcc': [
        "a dermoscopic image of basal cell carcinoma, arborizing vessels, blue-gray ovoid nests, maple leaf-like areas, spoke-wheel structures, shiny white lines, clinical quality dermoscopy",
        "a dermoscopic image of basal cell carcinoma, short fine telangiectasias, multiple small erosions, ulceration, blue-gray dots, crystalline structures, sharp focus dermoscopy",
        "a dermoscopic image of BCC skin lesion, tree-like branching vessels, white shiny areas, rosettes, milia-like cysts, well-defined border, high resolution clinical dermoscopy",
    ],
    'akiec': [
        "a dermoscopic image of actinic keratosis, strawberry pattern, rough scaly surface, rosette sign, white structureless areas, dotted vessels, clinical quality dermoscopy",
        "a dermoscopic image of actinic keratosis Bowen disease, red pseudo-network, surface scale, targetoid hair follicles, glomerular vessels, sharp focus clinical dermoscopy",
        "a dermoscopic image of intraepithelial carcinoma, keratinization, erythematous background, clustered dotted vessels, white halo around follicles, high magnification dermoscopy",
    ],
    'df': [
        "a dermoscopic image of dermatofibroma, central white scar-like patch, peripheral delicate pigment network, ring-like pattern, brown reticular lines at periphery, clinical quality dermoscopy",
        "a dermoscopic image of dermatofibroma skin lesion, white central area, faint peripheral pseudo-network, crystalline structures, shiny white lines, sharp focus dermoscopy",
        "a dermoscopic image of dermatofibroma, homogeneous white center, surrounding light brown pigment network, well-circumscribed border, smooth surface, clinical dermoscopy image",
    ],
    'vasc': [
        "a dermoscopic image of vascular lesion, red-purple lacunae, well-demarcated border, red-blue homogeneous areas, vascular spaces, clinical quality dermoscopy",
        "a dermoscopic image of vascular skin lesion, dark red to purple lagoons, whitish veil, thrombosed areas, sharply defined margin, high resolution dermoscopy",
        "a dermoscopic image of hemangioma, red-blue globular structures, lacunar pattern, well-circumscribed border, smooth surface, clinical quality dermoscopy image",
    ],
}

# prompt negative
CLASS_NEGATIVE_PROMPTS = {
    'mel': "benign nevus, symmetric, uniform color, regular border, blurry, low quality, artifacts, text, watermark, cartoon, non-dermoscopic, overexposed",
    'bcc': "melanoma, nevus, pigment network, blurry, low quality, artifacts, text, watermark, cartoon, non-dermoscopic, overexposed",
    'akiec': "melanoma, smooth surface, no scale, blurry, low quality, artifacts, text, watermark, cartoon, non-dermoscopic, overexposed",
    'df': "melanoma, irregular border, blue-white veil, blurry, low quality, artifacts, text, watermark, cartoon, non-dermoscopic, overexposed",
    'vasc': "melanoma, pigment network, brown color, blurry, low quality, artifacts, text, watermark, cartoon, non-dermoscopic, overexposed",
}

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class DermDiffusionDataset(Dataset):

    # inisiasi dataset dengan path image dan kelas
    def __init__(self, image_paths, class_name, resolution=512):
        self.image_paths = image_paths
        self.class_name = class_name
        self.resolution = resolution
        # preprocessing image
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
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = self.transform(img)
        # Rotate antara prompts yang tersedia
        prompts_pool = CLASS_PROMPTS.get(self.class_name,
            [f"a dermoscopic image of {self.class_name} skin lesion, clinical quality, high resolution"])
        prompt = prompts_pool[idx % len(prompts_pool)]
        return {"pixel_values": img, "prompt": prompt}

print("Dataset class ready")

# Sample Dataset (pair image and text prompt per class)

import matplotlib.pyplot as plt
cls = 'mel'
sample_paths = diffusion_data[cls][:6]
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
    ax.axis('off')

plt.tight_layout()
plt.show()

from huggingface_hub import login
login()

from huggingface_hub import whoami
print(whoami())

from diffusers import StableDiffusionPipeline, DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
import torch

MODEL_ID = "Manojb/stable-diffusion-2-1-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading tokenizer & text encoder...")
tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer", token=True)
text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder", torch_dtype=torch.float16, token=True)

print("Loading VAE...")
vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float16, token=True)

print("Loading UNet...")
unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet", torch_dtype=torch.float16, token=True)

print("Loading scheduler...")
noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler", token=True)

# Freeze VAE dan text encoder
vae.requires_grad_(False)
text_encoder.requires_grad_(False)

# LoRA config untuk UNet
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    init_lora_weights="gaussian",
    target_modules=[
        "to_k", "to_q", "to_v", "to_out.0",
        "proj_in", "proj_out",
        "ff.net.0.proj", "ff.net.2",
    ],
    lora_dropout=0.05,
)

unet = get_peft_model(unet, lora_config)
unet.print_trainable_parameters()

vae.to(DEVICE)
text_encoder.to(DEVICE)
unet.to(DEVICE)

print(f"\nStable Diffusion 2.1 + LoRA loaded on {DEVICE}!")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import gc

def train_lora_for_class(
    class_name,
    image_paths,
    unet,
    vae,
    text_encoder,
    tokenizer,
    noise_scheduler,
    num_epochs=100,
    batch_size=4,
    learning_rate=1e-4,
    gradient_accumulation_steps=4,
    save_dir=None,
):

    print(f"Training LoRA for class: {class_name.upper()}")
    print(f"  Images: {len(image_paths)}, Epochs: {num_epochs}, BS: {batch_size}")
    print(f"  Effective BS: {batch_size * gradient_accumulation_steps}")
    print(f"{'='*60}")

    dataset = DermDiffusionDataset(image_paths, class_name, resolution=512)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer hanya untuk LoRA parameters
    optimizer = AdamW(
        [p for p in unet.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=1e-2,
    )

    # LR scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs * len(dataloader)
    )

    scaler = GradScaler()
    unet.train()
    global_step = 0
    best_loss = float('inf')

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(dataloader):
            pixel_values = batch["pixel_values"].to(DEVICE, dtype=torch.float16)
            prompts = batch["prompt"]

            # Tokenize prompts
            text_inputs = tokenizer(
                prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(DEVICE)

            with autocast():
                # Encode text
                encoder_hidden_states = text_encoder(text_input_ids)[0]

                # Encode images ke latent space
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=DEVICE
                ).long()

                # Add noise ke latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Predict noise
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

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            if save_dir:
                best_path = os.path.join(save_dir, f"lora_{class_name}_best")
                unet.save_pretrained(best_path)

    # Save final
    if save_dir:
        final_path = os.path.join(save_dir, f"lora_{class_name}_final")
        unet.save_pretrained(final_path)
        print(f"Saved to {final_path}")

    print(f"Best loss: {best_loss:.6f}")

    return best_loss

print("Training function ready!")

import gc

LORA_SAVE_DIR = os.path.join(NOTEBOOK_DIR, 'temp/lora_weights')
os.makedirs(LORA_SAVE_DIR, exist_ok=True)

TRAIN_CONFIG = {
    'num_epochs': 50,
    'batch_size': 4,
    'learning_rate': 1e-4,
    'gradient_accumulation_steps': 4,
}

training_results = {}

for cls_name in MINORITY_CLASSES:
    # ── SKIP kalau sudah pernah training ──
    final_path = os.path.join(LORA_SAVE_DIR, f"lora_{cls_name}_final")
    if os.path.exists(final_path):
        print(f"{cls_name} sudah ada di Drive, SKIP!")
        continue

    if cls_name not in diffusion_data or len(diffusion_data[cls_name]) == 0:
        print(f"No data for {cls_name}, skipping...")
        continue

    from peft import get_peft_model
    unet_base = UNet2DConditionModel.from_pretrained(
        MODEL_ID, subfolder="unet", torch_dtype=torch.float16
    )
    unet = get_peft_model(unet_base, lora_config)
    unet.to(DEVICE)

    best_loss = train_lora_for_class(
        class_name=cls_name,
        image_paths=diffusion_data[cls_name],
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        noise_scheduler=noise_scheduler,
        save_dir=LORA_SAVE_DIR,
        **TRAIN_CONFIG,
    )

    training_results[cls_name] = best_loss

    del unet, unet_base
    gc.collect()
    torch.cuda.empty_cache()

print("\n" + "=" * 50)
print("TRAINING SUMMARY")
print("=" * 50)
for cls, loss in training_results.items():
    print(f"  {cls:10s}: best_loss = {loss:.6f}")