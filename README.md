# DermaDiff — SDXL LoRA Pipeline

This directory contains the DermaDiff Stable Diffusion XL + LoRA experiment
scripts, structured to match the repo convention used by the other generator
variants (e.g., SD 2.1 under `models/stable-diffusion-2.1-base/`).

## Repo Layout

```
dermadiff/
├── 0_dataset_prep.py              # Phase 0 — shared, builds HAM splits + per-class pool
├── dataset/                        # External dataset utilities (e.g., isic2019.py)
├── models/
│   ├── stable-diffusion-2.1-base/  # C2 — teammate's SD 2.1 LoRA
│   │   ├── fine_tuned_LoRA.py
│   │   ├── generate_images.py
│   │   ├── panderm_classifiers.py
│   │   ├── evaluation.py
│   │   └── LoRA Weights/
│   │       └── lora_{class}_final/
│   └── stable-diffusion-xl-base/   # C3 — this work (SDXL LoRA)
│       ├── fine_tuned_LoRA.py     # Phase 1 — LoRA fine-tuning
│       ├── generate_images.py     # Phase 2 — synthetic generation
│       ├── panderm_classifiers.py # Phase 3 — classifier training
│       ├── evaluation.py          # Phase 4 — test set evaluation
│       └── LoRA Weights/
│           └── lora_{class}_final/   # auto-created by Phase 1
├── README.md
└── requirements.txt
```

## Pipeline Overview

```
   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐   ┌──────────────────┐   ┌─────────────┐
   │ 0_dataset_prep  │ → │ fine_tuned_LoRA │ → │ generate_images │ → │ panderm_         │ → │ evaluation  │
   │                 │   │                 │   │                 │   │   classifiers    │   │             │
   │ HAM splits +    │   │ SDXL + LoRA     │   │ Synthetic       │   │ PanDerm ViT-L    │   │ Macro F1,   │
   │ per-class pool  │   │ (per class)     │   │ dermoscopic img │   │ (real + synth)   │   │ per-class   │
   └─────────────────┘   └─────────────────┘   └─────────────────┘   └──────────────────┘   └─────────────┘
        (root)                  models/stable-diffusion-xl-base/
```

## Setup

```bash
# 1. Install base Python dependencies
pip install -r requirements.txt

# 2. Clone diffusers at the exact version used for the published results
#    Phase 1 uses train_text_to_image_lora_sdxl.py from this repo.
#    The version check inside that script requires diffusers installed
#    FROM SOURCE (not from PyPI), so we pin to release tag v0.30.0
#    for reproducibility.
git clone --branch v0.30.0 https://github.com/huggingface/diffusers.git
pip install ./diffusers

# 3. Clone PanDerm (Phase 3 uses its run_class_finetuning.py script)
git clone https://github.com/SiyuanYan1/PanDerm.git
```

### Reproducibility Notes

**Why diffusers is pinned to `v0.30.0`:** The SDXL LoRA training script
(`train_text_to_image_lora_sdxl.py`) contains a `check_min_version()` call at
the top that requires diffusers be installed from source, not from PyPI. The
DermaDiff C3 results (Macro F1 = 0.8409) were produced using the
`train_text_to_image_lora_sdxl.py` script from diffusers `v0.30.0`. Using a
different version may produce slightly different LoRA weights, which could
affect downstream synthetic image quality and final classifier metrics.

**PanDerm PyTorch compatibility:** Phase 3 automatically patches PanDerm's
`run_class_finetuning.py` to add `weights_only=False` to its `torch.load()`
calls. This is required for PyTorch 2.6+ compatibility — without the patch,
loading the pretrained PanDerm checkpoint raises an exception because newer
PyTorch versions default `weights_only=True` for security. The patch is
idempotent and only applied once per checkout.

**PanDerm timm version lock:** `requirements.txt` pins `timm==0.9.16` because
newer timm versions break PanDerm's ViT loading code. This is a documented
PanDerm constraint, not specific to DermaDiff.

You also need the pretrained PanDerm checkpoint
(`panderm_ll_data6_checkpoint-499.pth`) — download manually from the PanDerm
GitHub release page: https://github.com/SiyuanYan1/PanDerm

## Dataset Access

DermaDiff uses three datasets, each with different access requirements. Only
HAM10000 is strictly required — ISIC 2019 and the longitudinal dataset are
optional augmentation sources that improve LoRA training quality when
available.

### 1. HAM10000 (required) — auto-downloadable

HAM10000 (Tschandl et al., 2018) is hosted on Harvard Dataverse under a
non-commercial research license and can be downloaded programmatically via
the Dataverse API:

```bash
python dataset/ham10000.py --output_dir ./data/ham10000
```

This script fetches `HAM10000_images_part_1.zip`, `HAM10000_images_part_2.zip`,
and the metadata file, then unpacks everything into the layout that Phase 0
expects: `data/ham10000/images/` (all 10,015 images flat) and
`data/ham10000/HAM10000_metadata.csv`. Download size is roughly 5.2 GB and
takes 10-20 minutes depending on your connection. The script is idempotent —
re-running it is a no-op once the data is in place.

Manual alternative: https://doi.org/10.7910/DVN/DBW86T (Harvard Dataverse).
Download the two image zips + metadata, unpack into the same layout above.

### 2. ISIC 2019 (optional) — auto-downloadable

Used as a supplementary source for LoRA training images. Your teammate's
`dataset/isic2019.py` script handles the download. See that script's
docstring for exact usage.

Phase 0 works without ISIC 2019 if you omit the `--isic_images` flag —
only HAM10000 train-split images will be used for LoRA training, which
produces weaker LoRAs for small classes (df, vasc) but is still functional.

### 3. Longitudinal Dataset (optional) — access-restricted, manual only

The longitudinal dermoscopic dataset used for the published DermaDiff C3
results originates from the University of Queensland (UQ) and is
access-restricted. **It cannot be redistributed in this repository**, and
there is no download script. If you want to reproduce the exact published
training pool:

1. Request access to the UQ longitudinal dermoscopic dataset through the
   original authors or your institution's data access process.
2. Download the full `dataset.zip` (~66 GB) from the UQ-provided source.
3. Extract it locally so that the expected directory layout looks like:
   ```
   data/longitudinal/
   └── dermoscopic_extracted/
       └── 866990d01449152d_NIMARE-A11453_A11453/
           └── data/
               └── Dermoscopic Images/
                   ├── HighRisk Dermoscopic images.xlsx
                   ├── General Dermosopic images.xlsx
                   └── <image files in nested subfolders>
   ```
4. Pass the extraction root as `--longitudinal_dir` and both Excel files
   as `--longitudinal_metadata` when running Phase 0.

**If you don't have longitudinal access**, skip this step entirely — omit
`--longitudinal_dir` and `--longitudinal_metadata` from the Phase 0
command. Phase 0 will gracefully fall back to using only HAM10000 (and
optionally ISIC 2019) for the per-class training pool. The resulting LoRAs
will train on fewer minority-class images, which may slightly reduce
downstream classifier performance — the published C3 Macro F1 = 0.8409 was
achieved with all three sources combined.

## Phase 0 — Dataset Preparation (root)

Builds the HAM10000 train/val/test splits and assembles the per-class image
pool for Phase 1. Critical filtering: HAM10000 contributes **train split only**,
ISIC 2019 and longitudinal contribute all images.

```bash
python 0_dataset_prep.py \
    --ham_images ./data/ham10000/images \
    --ham_metadata ./data/ham10000/HAM10000_metadata.csv \
    --isic_images ./data/isic2019/images \
    --longitudinal_dir ./data/longitudinal \
    --longitudinal_metadata \
        "./data/longitudinal/HighRisk Dermoscopic images.xlsx" \
        "./data/longitudinal/General Dermosopic images.xlsx" \
    --output_splits ./outputs/ham10000_splits.json \
    --output_per_class_dir ./outputs/training_images_per_class
```

Outputs:
- `outputs/ham10000_splits.json` — used by Phases 3 (auto train counts) and 3 (CSV building)
- `outputs/training_images_per_class/{mel,bcc,akiec,df,vasc}/` — used by Phase 1
- All images are **symlinked** (no disk duplication)

## Phase 1 — SDXL LoRA Fine-tuning

```bash
python models/stable-diffusion-xl-base/fine_tuned_LoRA.py \
    --train_data_dir ./outputs/training_images_per_class \
    --output_dir "./models/stable-diffusion-xl-base/LoRA Weights" \
    --diffusers_dir ./diffusers
```

LoRA weights are saved to:
```
models/stable-diffusion-xl-base/LoRA Weights/
├── lora_mel_final/pytorch_lora_weights.safetensors
├── lora_bcc_final/pytorch_lora_weights.safetensors
├── lora_akiec_final/pytorch_lora_weights.safetensors
├── lora_df_final/pytorch_lora_weights.safetensors
└── lora_vasc_final/pytorch_lora_weights.safetensors
```

Note: SDXL LoRA uses the `pytorch_lora_weights.safetensors` format (loaded
with `pipe.load_lora_weights()`), while the C2 SD 2.1 directory uses
`adapter_model.safetensors` (PEFT adapter format). The folder naming is
intentionally aligned with C2 for consistency, but the file formats inside
differ because the upstream training scripts produce different output formats.

Crash-safe: skips classes whose LoRA already exists.

## Phase 2 — Synthetic Image Generation

Auto mode (recommended — derives counts from Phase 0 splits):

```bash
python models/stable-diffusion-xl-base/generate_images.py \
    --lora_dir "./models/stable-diffusion-xl-base/LoRA Weights" \
    --output_dir ./outputs/synthetic_images \
    --splits_json ./outputs/ham10000_splits.json \
    --ham_metadata ./data/ham10000/HAM10000_metadata.csv \
    --ratio 2
```

Manual mode (if you don't want to use Phase 0 outputs):

```bash
python models/stable-diffusion-xl-base/generate_images.py \
    --lora_dir "./models/stable-diffusion-xl-base/LoRA Weights" \
    --output_dir ./outputs/synthetic_images \
    --train_counts mel=779 bcc=360 akiec=229 df=81 vasc=99 \
    --ratio 2
```

## Phase 3 — PanDerm Classifier Training

```bash
python models/stable-diffusion-xl-base/panderm_classifiers.py \
    --ham_images ./data/ham10000/images \
    --ham_metadata ./data/ham10000/HAM10000_metadata.csv \
    --splits_json ./outputs/ham10000_splits.json \
    --synthetic_dir ./outputs/synthetic_images \
    --panderm_dir ./PanDerm \
    --panderm_weights ./weights/panderm_pretrained.pth \
    --output_dir ./outputs/classifier_c3_1x \
    --ratio 1
```

## Phase 4 — Evaluation

```bash
python models/stable-diffusion-xl-base/evaluation.py \
    --checkpoint ./outputs/classifier_c3_1x/checkpoint-best.pth \
    --csv_path /tmp/dermadiff_classifier/ham10000_c3_1x.csv \
    --image_dir /tmp/dermadiff_classifier/images_c3_1x \
    --panderm_dir ./PanDerm \
    --output_dir ./outputs/eval_c3_1x \
    --label "C3-1x (SDXL)"
```

## Hyperparameters Reference

| Phase | Parameter | Value |
|---|---|---|
| 0 | Split ratio | 70/15/15 |
| 0 | Split seed | 42 |
| 1 | LoRA rank | 16 |
| 1 | Learning rate | 1e-4 |
| 1 | Resolution | 1024×1024 |
| 1 | Effective batch | 4 |
| 2 | Inference steps | 30 |
| 2 | Guidance scale | 7.5 |
| 2 | Resolution | 1024×1024 |
| 3 | PanDerm model | PanDerm_Large_FT |
| 3 | Layer decay | 0.65 |
| 3 | Drop path | 0.2 |
| 3 | Mixup / CutMix | 0.8 / 1.0 |
| 3 | Epochs | 50 (10 warmup) |
| 3 | Learning rate | 5e-4 |

## Hardware Requirements

- **Phase 0:** Any CPU (file I/O only)
- **Phase 1-3:** NVIDIA A100 40GB+
- **Phase 4:** Any GPU with 8GB+ VRAM
