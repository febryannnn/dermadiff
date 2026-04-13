# DermaDiff

DermaDiff investigates whether diffusion-based synthetic augmentation improves
classification of rare minority classes in the HAM10000 skin lesion dataset.
Multiple diffusion generators are compared by training a PanDerm ViT-Large
classifier on real + synthetic images and measuring Macro F1 across all 7
diagnostic classes.

Pre-trained adapter weights are bundled in the repo, so results can be
reproduced from the generation phase onward without GPU-intensive fine-tuning.

## Experiment Overview

| Experiment | Generator | Rank | Published Macro F1 |
|---|---|---|---|
| Exp A (baseline) | None (real HAM10000 only) | — | 0.8114 |
| Exp B | SD 2.1 LoRA (teammate's) | — | 0.8218 |
| **Exp C** | **SDXL LoRA** | **16** | **0.8409** |
| Exp D | SD 3.5 Large | — | 0.8482 |
| **Exp E** | **SDXL DoRA** | **8** | **0.8471** |

**Key finding:** Exp E matches Exp D's classification benefit with a 3x
smaller adapter (rank 8 vs rank 16) and a smaller base model (SDXL vs SD 3.5).
FID and downstream Macro F1 are inversely correlated — Exp D has the worst FID
(235.0) but the best classification benefit. LPIPS diversity is the strongest
predictor of downstream improvement, supporting the "diversity-over-fidelity"
hypothesis.

## Repo Layout

```
dermadiff/
├── 0_dataset_prep.py                          # Phase 0 — shared, builds HAM splits + per-class pool
├── dataset/
│   ├── ham10000.py                            # HAM10000 downloader (Harvard Dataverse API)
│   └── isic2019.py                            # ISIC 2019 downloader (teammate's)
├── models/
│   ├── stable-diffusion-2.1-base/             # Exp B — teammate's SD 2.1 LoRA
│   │   ├── fine_tuned_LoRA.py
│   │   ├── generate_images.py
│   │   ├── panderm_classifiers.py
│   │   ├── evaluation.py
│   │   └── LoRA Weights/
│   │       └── lora_{class}_final/
│   ├── stable-diffusion-xl-base/              # Exp C — SDXL LoRA (rank 16)
│   │   ├── fine_tuned_LoRA.py                 # Phase 1 — LoRA fine-tuning wrapper
│   │   ├── generate_images.py                 # Phase 2 — synthetic generation
│   │   ├── classifiers_training_LoRA.py        # Phase 3 — classifier training
│   │   ├── evaluation.py                      # Phase 4 — test set evaluation
│   │   ├── train_text_to_image_lora_sdxl.py   # Bundled from diffusers v0.37.1
│   │   └── LoRA Weights/                      # Pre-trained LoRAs (~440 MB)
│   │       └── lora_{mel,bcc,akiec,df,vasc}_final/pytorch_lora_weights.safetensors
│   └── stable-diffusion-xl-base-dora/         # Exp E — SDXL DoRA (rank 8)
│       ├── fine_tuned_DoRA.py                 # Phase 1 — DoRA fine-tuning wrapper
│       ├── generate_images.py                 # Phase 2 — synthetic generation
│       ├── classifiers_training_DoRA.py        # Phase 3 — shared logic with Exp C
│       ├── evaluation.py                      # Phase 4 — shared logic with Exp C
│       ├── train_dreambooth_lora_sdxl.py       # Bundled from diffusers v0.37.1
│       └── LoRA Weights/                      # Pre-trained DoRAs (~240 MB)
│           └── lora_{mel,bcc,akiec,df,vasc}_final/pytorch_lora_weights.safetensors
├── README.md
└── requirements.txt
```

## Pipeline Overview

```
   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐   ┌──────────────────┐   ┌─────────────┐
   │ 0_dataset_prep  │ → │ fine_tuned_*    │ → │ generate_images │ → │ panderm_         │ → │ evaluation  │
   │                 │   │                 │   │                 │   │   classifiers    │   │             │
   │ HAM splits +    │   │ SDXL + LoRA or  │   │ Synthetic       │   │ PanDerm ViT-L    │   │ Macro F1,   │
   │ per-class pool  │   │ DoRA (per class)│   │ dermoscopic img │   │ (real + synth)   │   │ per-class   │
   └─────────────────┘   └─────────────────┘   └─────────────────┘   └──────────────────┘   └─────────────┘
        (root)                  models/stable-diffusion-xl-base[-dora]/
```

Each experiment directory under `models/` follows the same Phase 1-4 structure.
Phase 3 and Phase 4 scripts are identical between Exp C and Exp E — the
classifier and evaluation logic consume images from a directory regardless of
which generator produced them.

## Setup

```bash
# 1. Install base Python dependencies
pip install -r requirements.txt

# 2. Clone PanDerm (Phase 3 uses its run_class_finetuning.py script)
git clone https://github.com/SiyuanYan1/PanDerm.git
```

Phase 1 training scripts are **bundled in this repo** from diffusers v0.37.1,
so you do not need to clone diffusers separately. If you prefer to use your
own diffusers checkout, pass `--diffusers_dir /path/to/diffusers` to the
Phase 1 wrapper — the bundled script is used as a fallback.

### Reproducibility Notes

**Bundled training scripts:** Both `train_text_to_image_lora_sdxl.py` (Exp C)
and `train_dreambooth_lora_sdxl.py` (Exp E) are committed directly from
diffusers v0.37.1. They are treated as read-only reference code — the Phase 1
wrappers (`fine_tuned_LoRA.py`, `fine_tuned_DoRA.py`) have a
`resolve_train_script()` helper that prefers these bundled scripts, with
`--diffusers_dir` as an optional override. This eliminates the need to install
diffusers from source.

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

Used as a supplementary source for fine-tuning images. Your teammate's
`dataset/isic2019.py` script handles the download. See that script's
docstring for exact usage.

Phase 0 works without ISIC 2019 if you omit the `--isic_images` flag —
only HAM10000 train-split images will be used for fine-tuning, which
produces weaker adapters for small classes (df, vasc) but is still functional.

### 3. Longitudinal Dataset (optional) — access-restricted, manual only

The longitudinal dermoscopic dataset used for the published DermaDiff Exp C
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
downstream classifier performance — the published Exp C Macro F1 = 0.8409 was
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
- `outputs/ham10000_splits.json` — used by Phases 2 (auto train counts) and 3 (CSV building)
- `outputs/training_images_per_class/{mel,bcc,akiec,df,vasc}/` — used by Phase 1
- All images are **symlinked** (no disk duplication)

## Phases 1-4 — Experiment Pipelines

After Phase 0, each experiment follows the same four-phase workflow
(fine-tune, generate, classify, evaluate). Pre-trained adapter weights are
bundled in the repo, so you can skip Phase 1 and start from Phase 2.

See the README inside each experiment directory for commands, hyperparameters,
and experiment-specific notes:

- **[Exp C — SDXL LoRA (rank 16)](models/stable-diffusion-xl-base/README.md)**
- **[Exp E — SDXL DoRA (rank 8)](models/stable-diffusion-xl-base-dora/README.md)**

Phase 3 (classifier) and Phase 4 (evaluation) scripts are identical between
Exp C and Exp E — the classifier and evaluation logic consume images from a
directory regardless of which generator produced them.

## Hardware Requirements

- **Phase 0:** Any CPU (file I/O only)
- **Phase 1-2:** NVIDIA A100 40GB+ (or T4 with reduced batch)
- **Phase 3:** NVIDIA A100 40GB+
- **Phase 4:** Any GPU with 8GB+ VRAM
