# Evaluation — Cross-Model Quality & Cross-Domain Generalization

This directory contains two evaluation scripts that operate across all
experiments, independent of any single generator.

## Scripts

| Script | Purpose |
|---|---|
| `image_quality_metrics.py` | FID, LPIPS diversity, MS-SSIM for synthetic images |
| `cross_domain_eval.py` | OOD evaluation of trained classifiers on PAD-UFES-20 |

## Dependencies

In addition to the base `requirements.txt`, these scripts require:

```bash
pip install torchmetrics[image] lpips pytorch-msssim scikit-learn matplotlib pandas
```

`cross_domain_eval.py` also requires a cloned PanDerm repository (for model
loading). See the root README for setup instructions.

---

## image_quality_metrics.py

Computes three metrics comparing synthetic images against real HAM10000
images, per class (mel, bcc, akiec, df, vasc) and aggregated as mean.

### Metrics

| Metric | What it measures | Model / Method | Preprocessing | Interpretation |
|---|---|---|---|---|
| FID | Realism (distribution distance) | InceptionV3, 2048-dim pooling | 299x299, [0,1] | Lower = better |
| LPIPS diversity | Intra-class variety of synthetics | AlexNet backbone | 256x256, [-1,1] | Higher = more diverse |
| MS-SSIM | Structural similarity to real images | Multi-scale SSIM | 256x256, [0,1] | Higher = more similar |

- LPIPS samples **500 random pairs within synthetic images** of the same class
- MS-SSIM samples **500 random real-synthetic pairs** per class
- All sampling uses `seed=42` for reproducibility

### Directory structure expected

Both `--real_dir` and each experiment path must be organized as per-class
subdirectories:

```
images_classified/          synthetic_images_expc/
├── mel/                    ├── mel/
├── bcc/                    ├── bcc/
├── akiec/                  ├── akiec/
├── df/                     ├── df/
└── vasc/                   └── vasc/
```

### Usage

Full run (all 4 experiments, all metrics):

```bash
python evaluation/image_quality_metrics.py \
    --real_dir ./data/ham10000/images_classified \
    --experiments \
        "Exp B" ./outputs/synthetic_images_expb \
        "Exp C" ./outputs/synthetic_images_expc \
        "Exp D" ./outputs/synthetic_images_expd \
        "Exp E" ./outputs/synthetic_images_expe \
    --output_dir ./outputs/image_quality_metrics
```

Single experiment, skip FID (faster):

```bash
python evaluation/image_quality_metrics.py \
    --real_dir ./data/ham10000/images_classified \
    --experiments "Exp E" ./outputs/synthetic_images_expe \
    --output_dir ./outputs/image_quality_metrics \
    --skip_fid
```

### Arguments

| Argument | Required | Description |
|---|---|---|
| `--real_dir` | Yes | Real images with per-class subdirectories |
| `--experiments` | Yes | Pairs of `"Name" path` for each experiment |
| `--output_dir` | Yes | Where to write JSON + CSV results |
| `--classes` | No | Classes to evaluate (default: mel bcc akiec df vasc) |
| `--n_pairs` | No | Number of pairs for LPIPS/MS-SSIM (default: 500) |
| `--skip_fid` | No | Skip FID computation |
| `--skip_lpips` | No | Skip LPIPS computation |
| `--skip_msssim` | No | Skip MS-SSIM computation |
| `--device` | No | `cuda` or `cpu` (default: auto-detect) |

### Outputs

| File | Contents |
|---|---|
| `image_quality_metrics.json` | Nested dict: `metric -> experiment -> class -> score` |
| `image_quality_metrics.csv` | Flat table: one row per experiment-class pair + mean rows |

---

## cross_domain_eval.py

Evaluates trained PanDerm ViT-L classifiers on PAD-UFES-20 to test
out-of-distribution generalization. The classifiers were trained on
HAM10000 (7 classes); PAD-UFES-20 has 5 classes (akiec, bcc, bkl, mel, nv)
with no df or vasc samples.

### PAD-UFES-20 dataset

- 2,106 clinical (non-dermoscopic) images
- Requires a CSV file with `img_id` and `dx` columns
- Images are found recursively under `--pad_images` (supports nested
  subdirectories like `imgs_part_1/`, `imgs_part_2/`, etc.)

### Usage

```bash
python evaluation/cross_domain_eval.py \
    --pad_csv ./data/pad-ufes-20/pad_ufes_mapped.csv \
    --pad_images ./data/pad-ufes-20/images \
    --panderm_dir ./PanDerm \
    --checkpoints \
        "Exp A" ./outputs/classifier_expa/checkpoint-best.pth \
        "Exp B" ./outputs/classifier_expb/checkpoint-best.pth \
        "Exp C" ./outputs/classifier_expc/checkpoint-best.pth \
        "Exp D" ./outputs/classifier_expd/checkpoint-best.pth \
        "Exp E" ./outputs/classifier_expe/checkpoint-best.pth \
    --output_dir ./outputs/cross_domain_eval
```

### Arguments

| Argument | Required | Description |
|---|---|---|
| `--pad_csv` | Yes | CSV with `img_id` and `dx` columns |
| `--pad_images` | Yes | Root directory with PAD-UFES-20 images (recursive search) |
| `--panderm_dir` | Yes | Path to cloned PanDerm repository |
| `--checkpoints` | Yes | Pairs of `"Name" path` for each checkpoint |
| `--output_dir` | Yes | Where to write results |
| `--batch_size` | No | Inference batch size (default: 64) |
| `--device` | No | `cuda` or `cpu` (default: auto-detect) |

### Outputs

| File | Contents |
|---|---|
| `cross_domain_results.csv` | Summary: accuracy, weighted/macro F1, per-class recall per experiment |
| `predictions_{experiment}.csv` | Per-image predictions: img_id, true label, predicted label, correct |
| `confusion_matrices.png` | Normalized confusion matrices (one per experiment) |
| `per_class_recall.png` | Grouped bar chart of per-class recall across experiments |

### Classifier loading

The script handles multiple checkpoint formats (keys: `model`,
`model_state_dict`, `state_dict`, or raw state dict) and automatically
strips `module.` prefixes from DataParallel-wrapped checkpoints. Missing
checkpoints are skipped gracefully.
