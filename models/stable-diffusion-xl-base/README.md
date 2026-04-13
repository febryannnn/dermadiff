# Exp C — SDXL LoRA (rank 16)

SDXL + LoRA fine-tuning for minority-class skin lesion generation. This
experiment uses `train_text_to_image_lora_sdxl.py` (text-to-image training
with metadata.jsonl) from diffusers v0.37.1, bundled in this directory.

**Published result:** Macro F1 = 0.8409

## Phase 1 — LoRA Fine-tuning

> **Skip Phase 1 — Pre-trained LoRAs are included in this repo**
>
> The five SDXL LoRA adapters used for the published results are bundled at
> `LoRA Weights/lora_{class}_final/` (~89 MB each, ~440 MB total). To
> reproduce results, skip to Phase 2 — generation uses the bundled weights
> by default.
>
> Phase 1 is only needed if you want to retrain from scratch (e.g., on a
> different dataset or with different hyperparameters).

```bash
python models/stable-diffusion-xl-base/fine_tuned_LoRA.py \
    --train_data_dir ./outputs/training_images_per_class \
    --output_dir "./models/stable-diffusion-xl-base/LoRA Weights"
```

The wrapper uses `resolve_train_script()` to locate the bundled
`train_text_to_image_lora_sdxl.py`. Pass `--diffusers_dir /path/to/diffusers`
to override with your own checkout.

Crash-safe: skips classes whose weights already exist.

Note: Exp B (SD 2.1) uses `adapter_model.safetensors` (PEFT adapter format)
— different from the `pytorch_lora_weights.safetensors` format used here.
The folder naming is consistent across experiments, but the file formats
differ because the upstream training scripts produce different outputs.

## Phase 2 — Synthetic Image Generation

Bundled weights are loaded automatically — `--lora_dir` defaults to the
`LoRA Weights/` directory next to the script.

Auto mode (recommended — derives counts from Phase 0 splits):

```bash
python models/stable-diffusion-xl-base/generate_images.py \
    --output_dir ./outputs/synthetic_images_expc \
    --splits_json ./outputs/ham10000_splits.json \
    --ham_metadata ./data/ham10000/HAM10000_metadata.csv \
    --ratio 2
```

Manual mode (specify counts directly):

```bash
python models/stable-diffusion-xl-base/generate_images.py \
    --output_dir ./outputs/synthetic_images_expc \
    --train_counts mel=779 bcc=360 akiec=229 df=81 vasc=99 \
    --ratio 2
```

## Phase 3 — PanDerm Classifier Training

```bash
python models/stable-diffusion-xl-base/classifiers_training_LoRA.py \
    --ham_images ./data/ham10000/images \
    --ham_metadata ./data/ham10000/HAM10000_metadata.csv \
    --splits_json ./outputs/ham10000_splits.json \
    --synthetic_dir ./outputs/synthetic_images_expc \
    --panderm_dir ./PanDerm \
    --panderm_weights ./weights/panderm_pretrained.pth \
    --output_dir ./outputs/classifier_expc \
    --ratio 1
```

## Phase 4 — Evaluation

```bash
python models/stable-diffusion-xl-base/evaluation.py \
    --checkpoint ./outputs/classifier_expc/checkpoint-best.pth \
    --csv_path ./outputs/eval_expc/ham10000_expc.csv \
    --image_dir ./outputs/eval_expc/images \
    --panderm_dir ./PanDerm \
    --output_dir ./outputs/eval_expc \
    --label "Exp C (SDXL LoRA)"
```

## Hyperparameters

### Phase 1 — Fine-tuning

| Parameter | Value |
|---|---|
| Training script | `train_text_to_image_lora_sdxl.py` |
| LoRA rank | 16 |
| Learning rate | 1e-4 |
| LR scheduler | constant |
| LoRA dropout | 0 |
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
