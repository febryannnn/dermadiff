#!/usr/bin/env python3
"""
DermaDiff — Cross-Model Image Quality Metrics
==============================================

Computes FID, LPIPS diversity, and MS-SSIM for synthetic images produced by
each generator experiment, compared against real HAM10000 images.

Metrics
-------
- **FID** (Frechet Inception Distance): lower = more realistic.
  Uses torchmetrics InceptionV3 at 2048-dim pooling layer.
- **LPIPS diversity**: higher = more diverse within a class.
  Uses AlexNet backbone, 500 random intra-class synthetic pairs.
- **MS-SSIM** (Multi-Scale Structural Similarity): higher = more similar
  to real images. 500 random real-synthetic pairs per class.

All metrics are computed per class (mel, bcc, akiec, df, vasc) and
aggregated as mean across classes.

Usage
-----
    python evaluation/image_quality_metrics.py \\
        --real_dir ./data/ham10000/images_classified \\
        --experiments \\
            "Exp B" ./outputs/synthetic_images_expb \\
            "Exp C" ./outputs/synthetic_images_expc \\
            "Exp D" ./outputs/synthetic_images_expd \\
            "Exp E" ./outputs/synthetic_images_expe \\
        --output_dir ./outputs/image_quality_metrics

    Each synthetic directory must contain per-class subdirectories:
        synthetic_images_expc/
        ├── mel/
        ├── bcc/
        ├── akiec/
        ├── df/
        └── vasc/

    The real_dir must also have per-class subdirectories (at least the
    5 target classes above).
"""

import argparse
import csv
import json
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_CLASSES = ["mel", "bcc", "akiec", "df", "vasc"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SEED = 42
N_PAIRS = 500
BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _list_images(directory):
    """Return sorted list of image paths in *directory*."""
    paths = []
    for fname in os.listdir(directory):
        if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
            paths.append(os.path.join(directory, fname))
    paths.sort()
    return paths


class _ImageDataset(Dataset):
    """Loads images from a flat list of paths with a given transform."""

    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


# ---------------------------------------------------------------------------
# Metric: FID
# ---------------------------------------------------------------------------

def compute_fid(real_dir, synth_dir, device):
    """Compute FID between real and synthetic images for one class."""
    from torchmetrics.image.fid import FrechetInceptionDistance

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    real_paths = _list_images(real_dir)
    synth_paths = _list_images(synth_dir)
    if not real_paths or not synth_paths:
        return float("nan")

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    real_loader = DataLoader(
        _ImageDataset(real_paths, transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
    )
    synth_loader = DataLoader(
        _ImageDataset(synth_paths, transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
    )

    for batch in real_loader:
        fid.update(batch.to(device), real=True)
    for batch in synth_loader:
        fid.update(batch.to(device), real=False)

    score = fid.compute().item()
    del fid
    torch.cuda.empty_cache()
    return score


# ---------------------------------------------------------------------------
# Metric: LPIPS diversity
# ---------------------------------------------------------------------------

def compute_lpips_diversity(synth_dir, device):
    """Mean pairwise LPIPS among synthetic images (intra-class diversity)."""
    import lpips

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    synth_paths = _list_images(synth_dir)
    n = len(synth_paths)
    if n < 2:
        return float("nan")

    dataset = _ImageDataset(synth_paths, transform)
    lpips_model = lpips.LPIPS(net="alex").to(device)
    lpips_model.eval()

    random.seed(SEED)
    max_pairs = n * (n - 1) // 2
    num_pairs = min(N_PAIRS, max_pairs)

    pairs = set()
    while len(pairs) < num_pairs:
        a, b = random.sample(range(n), 2)
        pair = (min(a, b), max(a, b))
        pairs.add(pair)
    pairs = list(pairs)

    distances = []
    for i in range(0, len(pairs), BATCH_SIZE):
        batch_pairs = pairs[i : i + BATCH_SIZE]
        imgs_a = torch.stack([dataset[a] for a, _ in batch_pairs]).to(device)
        imgs_b = torch.stack([dataset[b] for _, b in batch_pairs]).to(device)
        # LPIPS expects [-1, 1]
        imgs_a = imgs_a * 2 - 1
        imgs_b = imgs_b * 2 - 1
        with torch.no_grad():
            d = lpips_model(imgs_a, imgs_b)
        distances.extend(d.squeeze().cpu().tolist() if d.dim() > 1
                         else [d.item()] if d.dim() == 0
                         else d.cpu().tolist())

    del lpips_model
    torch.cuda.empty_cache()
    return float(np.mean(distances))


# ---------------------------------------------------------------------------
# Metric: MS-SSIM
# ---------------------------------------------------------------------------

def compute_msssim(real_dir, synth_dir, device):
    """Mean MS-SSIM between random real-synthetic pairs."""
    from pytorch_msssim import ms_ssim

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    real_paths = _list_images(real_dir)
    synth_paths = _list_images(synth_dir)
    if not real_paths or not synth_paths:
        return float("nan")

    real_ds = _ImageDataset(real_paths, transform)
    synth_ds = _ImageDataset(synth_paths, transform)

    random.seed(SEED)
    num_pairs = min(N_PAIRS, len(real_paths), len(synth_paths))
    real_indices = [random.randint(0, len(real_ds) - 1) for _ in range(num_pairs)]
    synth_indices = [random.randint(0, len(synth_ds) - 1) for _ in range(num_pairs)]

    scores = []
    for ri, si in zip(real_indices, synth_indices):
        img_r = real_ds[ri].unsqueeze(0).to(device)
        img_s = synth_ds[si].unsqueeze(0).to(device)
        score = ms_ssim(img_r, img_s, data_range=1.0, size_average=True)
        scores.append(score.item())

    torch.cuda.empty_cache()
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute FID, LPIPS diversity, and MS-SSIM for synthetic images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--real_dir", required=True,
        help="Directory with real images organized in per-class subdirectories.",
    )
    parser.add_argument(
        "--experiments", required=True, nargs="+",
        help=(
            "Pairs of (name, path) for each experiment. "
            'E.g.: "Exp C" ./synth_expc "Exp E" ./synth_expe'
        ),
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory to write JSON and CSV results.",
    )
    parser.add_argument(
        "--classes", nargs="+", default=TARGET_CLASSES,
        help="Classes to evaluate (default: mel bcc akiec df vasc).",
    )
    parser.add_argument(
        "--n_pairs", type=int, default=N_PAIRS,
        help="Number of pairs for LPIPS and MS-SSIM (default: 500).",
    )
    parser.add_argument(
        "--skip_fid", action="store_true",
        help="Skip FID computation (fastest metric to skip).",
    )
    parser.add_argument(
        "--skip_lpips", action="store_true",
        help="Skip LPIPS computation.",
    )
    parser.add_argument(
        "--skip_msssim", action="store_true",
        help="Skip MS-SSIM computation.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: auto-detect).",
    )
    return parser.parse_args()


def _parse_experiments(raw):
    """Parse flat list ['name1', 'path1', 'name2', 'path2', ...] into pairs."""
    if len(raw) % 2 != 0:
        raise ValueError(
            "--experiments must be pairs of (name, path). Got odd number of arguments."
        )
    experiments = []
    for i in range(0, len(raw), 2):
        experiments.append((raw[i], raw[i + 1]))
    return experiments


def main():
    args = parse_args()
    global N_PAIRS
    N_PAIRS = args.n_pairs

    experiments = _parse_experiments(args.experiments)
    device = torch.device(args.device)
    classes = args.classes
    os.makedirs(args.output_dir, exist_ok=True)

    results = {"fid": {}, "lpips_diversity": {}, "ms_ssim": {}}

    for exp_name, exp_dir in experiments:
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_name}  ({exp_dir})")
        print(f"{'='*60}")

        results["fid"][exp_name] = {}
        results["lpips_diversity"][exp_name] = {}
        results["ms_ssim"][exp_name] = {}

        for cls in classes:
            real_cls_dir = os.path.join(args.real_dir, cls)
            synth_cls_dir = os.path.join(exp_dir, cls)

            if not os.path.isdir(synth_cls_dir):
                print(f"  [{cls}] SKIP — synthetic directory not found: {synth_cls_dir}")
                results["fid"][exp_name][cls] = None
                results["lpips_diversity"][exp_name][cls] = None
                results["ms_ssim"][exp_name][cls] = None
                continue

            if not os.path.isdir(real_cls_dir):
                print(f"  [{cls}] SKIP — real directory not found: {real_cls_dir}")
                results["fid"][exp_name][cls] = None
                results["lpips_diversity"][exp_name][cls] = None
                results["ms_ssim"][exp_name][cls] = None
                continue

            n_real = len(_list_images(real_cls_dir))
            n_synth = len(_list_images(synth_cls_dir))
            print(f"\n  [{cls}] real={n_real}, synthetic={n_synth}")

            # FID
            if not args.skip_fid:
                print(f"    Computing FID ...", end=" ", flush=True)
                fid_score = compute_fid(real_cls_dir, synth_cls_dir, device)
                results["fid"][exp_name][cls] = fid_score
                print(f"{fid_score:.1f}")
            else:
                results["fid"][exp_name][cls] = None

            # LPIPS
            if not args.skip_lpips:
                print(f"    Computing LPIPS diversity ...", end=" ", flush=True)
                lpips_score = compute_lpips_diversity(synth_cls_dir, device)
                results["lpips_diversity"][exp_name][cls] = lpips_score
                print(f"{lpips_score:.4f}")
            else:
                results["lpips_diversity"][exp_name][cls] = None

            # MS-SSIM
            if not args.skip_msssim:
                print(f"    Computing MS-SSIM ...", end=" ", flush=True)
                msssim_score = compute_msssim(real_cls_dir, synth_cls_dir, device)
                results["ms_ssim"][exp_name][cls] = msssim_score
                print(f"{msssim_score:.4f}")
            else:
                results["ms_ssim"][exp_name][cls] = None

    # -----------------------------------------------------------------------
    # Aggregate means
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Aggregate (mean across classes)")
    print(f"{'='*60}\n")

    header = f"{'Experiment':<20} {'FID':>8} {'LPIPS':>8} {'MS-SSIM':>8}"
    print(header)
    print("-" * len(header))

    for exp_name, _ in experiments:
        means = {}
        for metric in ["fid", "lpips_diversity", "ms_ssim"]:
            vals = [
                results[metric][exp_name].get(c)
                for c in classes
                if results[metric][exp_name].get(c) is not None
                and not (isinstance(results[metric][exp_name].get(c), float)
                         and np.isnan(results[metric][exp_name][c]))
            ]
            means[metric] = float(np.mean(vals)) if vals else None

        results["fid"].setdefault("_mean", {})[exp_name] = means["fid"]
        results["lpips_diversity"].setdefault("_mean", {})[exp_name] = means["lpips_diversity"]
        results["ms_ssim"].setdefault("_mean", {})[exp_name] = means["ms_ssim"]

        fid_str = f"{means['fid']:.1f}" if means["fid"] is not None else "N/A"
        lpips_str = f"{means['lpips_diversity']:.4f}" if means["lpips_diversity"] is not None else "N/A"
        msssim_str = f"{means['ms_ssim']:.4f}" if means["ms_ssim"] is not None else "N/A"
        print(f"{exp_name:<20} {fid_str:>8} {lpips_str:>8} {msssim_str:>8}")

    # -----------------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------------
    json_path = os.path.join(args.output_dir, "image_quality_metrics.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {json_path}")

    # -----------------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------------
    csv_path = os.path.join(args.output_dir, "image_quality_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "class", "fid", "lpips_diversity", "ms_ssim"])
        for exp_name, _ in experiments:
            for cls in classes:
                writer.writerow([
                    exp_name,
                    cls,
                    results["fid"][exp_name].get(cls, ""),
                    results["lpips_diversity"][exp_name].get(cls, ""),
                    results["ms_ssim"][exp_name].get(cls, ""),
                ])
            # Mean row
            writer.writerow([
                exp_name,
                "mean",
                results["fid"]["_mean"].get(exp_name, ""),
                results["lpips_diversity"]["_mean"].get(exp_name, ""),
                results["ms_ssim"]["_mean"].get(exp_name, ""),
            ])
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
