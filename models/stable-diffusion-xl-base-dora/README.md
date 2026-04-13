# Exp E — SDXL DoRA (rank 8)

SDXL + DoRA fine-tuning for minority-class skin lesion generation. DoRA
(Weight-Decomposed Low-Rank Adaptation, Liu et al. ICML 2024) decomposes
weight updates into magnitude and direction components, achieving competitive
results at lower rank than standard LoRA.

This experiment uses `train_dreambooth_lora_sdxl.py --use_dora` (DreamBooth
with per-class instance prompts) from diffusers v0.37.1, bundled in this
directory.

**Published result:** Macro F1 = 0.8471 — matches Exp D (SD 3.5 Large,
0.8482) with a 3x smaller adapter and a smaller base model.

## Phase 1 — DoRA Fine-tuning

> **Skip Phase 1 — Pre-trained DoRAs are included in this repo**
>
> The five SDXL DoRA adapters used for the published results are bundled at
> `LoRA Weights/lora_{class}_final/` (~48 MB each, ~240 MB total). To
> reproduce results, skip to Phase 2 — generation uses the bundled weights
> by default.
>
> Phase 1 is only needed if you want to retrain from scratch (e.g., on a
> different dataset or with different hyperparameters).

```bash
python models/stable-diffusion-xl-base-dora/fine_tuned_DoRA.py \
    --train_data_dir ./outputs/training_images_per_class \
    --output_dir "./models/stable-diffusion-xl-base-dora/LoRA Weights"
```

The wrapper uses `resolve_train_script()` to locate the bundled
`train_dreambooth_lora_sdxl.py`. Pass `--diffusers_dir /path/to/diffusers`
to override with your own checkout.

Key differences from Exp C: rank 8 (vs 16), lower learning rate with cosine
schedule, and dropout 0.05 — following NVIDIA DoRA paper recommendations.

Crash-safe: skips classes whose weights already exist.

## Phase 2 — Synthetic Image Generation

Bundled weights are loaded automatically — `--lora_dir` defaults to the
`LoRA Weights/` directory next to the script.

Auto mode (recommended — derives counts from Phase 0 splits):

```bash
python models/stable-diffusion-xl-base-dora/generate_images.py \
    --output_dir ./outputs/synthetic_images_expe \
    --splits_json ./outputs/ham10000_splits.json \
    --ham_metadata ./data/ham10000/HAM10000_metadata.csv \
    --ratio 2
```

Manual mode (specify counts directly):

```bash
python models/stable-diffusion-xl-base-dora/generate_images.py \
    --output_dir ./outputs/synthetic_images_expe \
    --train_counts mel=779 bcc=360 akiec=229 df=81 vasc=99 \
    --ratio 2
```

## Phase 3 — PanDerm Classifier Training

This script is identical to the Exp C version — the classifier trains on
real + synthetic images regardless of which generator produced them.

```bash
python models/stable-diffusion-xl-base-dora/panderm_classifiers.py \
    --ham_images ./data/ham10000/images \
    --ham_metadata ./data/ham10000/HAM10000_metadata.csv \
    --splits_json ./outputs/ham10000_splits.json \
    --synthetic_dir ./outputs/synthetic_images_expe \
    --panderm_dir ./PanDerm \
    --panderm_weights ./weights/panderm_pretrained.pth \
    --output_dir ./outputs/classifier_expe \
    --ratio 1
```

## Phase 4 — Evaluation

This script is identical to the Exp C version.

```bash
python models/stable-diffusion-xl-base-dora/evaluation.py \
    --checkpoint ./outputs/classifier_expe/checkpoint-best.pth \
    --csv_path ./outputs/eval_expe/ham10000_expe.csv \
    --image_dir ./outputs/eval_expe/images \
    --panderm_dir ./PanDerm \
    --output_dir ./outputs/eval_expe \
    --label "Exp E (SDXL DoRA)"
```

## Hyperparameters

### Phase 1 — Fine-tuning

| Parameter | Value |
|---|---|
| Training script | `train_dreambooth_lora_sdxl.py --use_dora` |
| LoRA rank | 8 |
| Learning rate | 5e-5 |
| LR scheduler | cosine |
| LR warmup steps | 100 |
| LoRA dropout | 0.05 |
| Resolution | 1024x1024 |
| Batch size | 1 |
| Gradient accumulation | 4 |
| Mixed precision | fp16 |
| Adaptive steps | <150 img: 1500, <400: 1000, else: 800 |

### Phase 2 — Generation

| Parameter | Value |
|---|---|
| Inference steps | 30 |
| Guidance scale | 7.5 |
| Resolution | 1024x1024 |

### Phase 3 — Classifier

| Parameter | Value |
|---|---|
| PanDerm model | PanDerm_Large_FT |
| Batch size | 128 |
| Learning rate | 5e-4 |
| Layer decay | 0.65 |
| Drop path | 0.2 |
| Mixup / CutMix | 0.8 / 1.0 |
| Epochs | 50 (10 warmup) |
