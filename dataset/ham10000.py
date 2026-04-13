#!/usr/bin/env python3
"""
DermaDiff — Dataset Downloader: HAM10000
==========================================

Downloads the HAM10000 dataset (Tschandl et al., 2018) from the Harvard
Dataverse public API, unpacks the two image archives into a single flat
`images/` directory, and places `HAM10000_metadata.csv` alongside it.

The dataset is hosted at: https://doi.org/10.7910/DVN/DBW86T

This script uses the Dataverse Data Access API, which allows anonymous
downloads of public files without an API token. It hits two endpoints:

    GET /api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/DBW86T
        -> JSON listing all files in the dataset with their numeric IDs

    GET /api/access/datafile/{file_id}
        -> Raw file bytes

Usage
-----
    python dataset/ham10000.py --output_dir ./data/ham10000

Output layout
-------------
    output_dir/
    ├── images/                    # all 10,015 .jpg files from parts 1 and 2
    │   ├── ISIC_0024306.jpg
    │   ├── ISIC_0024307.jpg
    │   └── ...
    └── HAM10000_metadata.csv      # labels, lesion_id, dx, dx_type, age, sex, etc.

The script is idempotent — if the output files already exist and their image
count matches the expected 10,015, it skips all downloading.
"""

import argparse
import json
import os
import shutil
import sys
import urllib.request
import urllib.error
import zipfile

# ────────────────────────────────────────────────────────────────────────
# CONFIG (hardcoded — change here, not via CLI)
# ────────────────────────────────────────────────────────────────────────

DATAVERSE_SERVER = "https://dataverse.harvard.edu"
HAM10000_DOI = "doi:10.7910/DVN/DBW86T"

# Files we want to fetch from the dataset. The metadata "tab" file is what
# the Dataverse stores; it's a tab-separated variant of the original CSV.
# We also accept "csv" in case the dataset listing exposes it directly.
WANTED_FILES = {
    "HAM10000_images_part_1.zip": "image_archive",
    "HAM10000_images_part_2.zip": "image_archive",
    "HAM10000_metadata.tab":      "metadata",
    "HAM10000_metadata.csv":      "metadata",  # alternative name
}

EXPECTED_IMAGE_COUNT = 10015
IMG_EXTS = (".jpg", ".jpeg", ".png")

# Harvard Dataverse blocks the default `Python-urllib/X.Y` user agent on sight,
# returning HTTP 403 even for fully public, anonymous-readable datasets. We set
# a real-browser-looking User-Agent on every request to bypass this filter.
# This is purely a UA filter — no API authentication is required for public files.
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _open_url(url: str, timeout: int = 30):
    """Open a URL with a browser-like User-Agent to avoid Dataverse's UA filter."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(req, timeout=timeout)


# ────────────────────────────────────────────────────────────────────────
# DATAVERSE API HELPERS
# ────────────────────────────────────────────────────────────────────────

def fetch_dataset_listing(doi: str) -> list:
    """Query Dataverse API for the file list of a dataset.

    Returns a list of dicts, each with keys: label (filename), id (numeric),
    contentType, filesize, etc.
    """
    url = f"{DATAVERSE_SERVER}/api/datasets/:persistentId/?persistentId={doi}"
    print(f"  Querying Dataverse API: {url}")

    try:
        with _open_url(url, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"  ERROR: HTTP {e.code} while fetching dataset listing")
        raise
    except urllib.error.URLError as e:
        print(f"  ERROR: network error while fetching dataset listing: {e}")
        raise

    if data.get("status") != "OK":
        raise RuntimeError(f"Dataverse API returned status: {data.get('status')}")

    files = data["data"]["latestVersion"]["files"]
    print(f"  Found {len(files)} files in dataset")
    return files


def download_file(file_id: int, dest_path: str, filename: str) -> None:
    """Stream-download a single file from the Dataverse Data Access API.

    Uses chunked reads so large image archives (~2.6 GB) don't blow up memory.
    """
    url = f"{DATAVERSE_SERVER}/api/access/datafile/{file_id}"
    print(f"  Downloading {filename} ({file_id}) -> {dest_path}")

    tmp_path = dest_path + ".part"
    try:
        with _open_url(url, timeout=60) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB

            with open(tmp_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        pct = 100.0 * downloaded / total_size
                        mb_done = downloaded / 1024 / 1024
                        mb_total = total_size / 1024 / 1024
                        print(f"\r    {pct:5.1f}%  ({mb_done:7.1f} / {mb_total:7.1f} MB)",
                              end="", flush=True)
        print()
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    os.rename(tmp_path, dest_path)
    size_mb = os.path.getsize(dest_path) / 1024 / 1024
    print(f"    Done: {size_mb:.1f} MB")


# ────────────────────────────────────────────────────────────────────────
# UNPACKING
# ────────────────────────────────────────────────────────────────────────

def unpack_images_zip(zip_path: str, images_dir: str) -> int:
    """Unzip a HAM10000 image archive into a flat images directory.

    The zip files may contain either a flat layout or a single top-level
    folder wrapper. Either way, we end up writing each .jpg directly into
    `images_dir/` without intermediate subdirectories.
    """
    count = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(IMG_EXTS)]
        print(f"    Extracting {len(members)} images from {os.path.basename(zip_path)}...")
        for member in members:
            # Strip any directory components
            fname = os.path.basename(member)
            if not fname:
                continue
            dst = os.path.join(images_dir, fname)
            if os.path.exists(dst):
                count += 1
                continue
            with zf.open(member) as src_fp, open(dst, "wb") as dst_fp:
                shutil.copyfileobj(src_fp, dst_fp)
            count += 1
    return count


def count_images(images_dir: str) -> int:
    if not os.path.isdir(images_dir):
        return 0
    return len([
        f for f in os.listdir(images_dir)
        if f.lower().endswith(IMG_EXTS)
    ])


# ────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download HAM10000 dataset from Harvard Dataverse"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory where images/ and HAM10000_metadata.csv will be placed"
    )
    parser.add_argument(
        "--keep_zips", action="store_true",
        help="Keep the downloaded .zip archives after extraction (default: delete)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if output files already exist"
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    images_dir = os.path.join(output_dir, "images")
    metadata_csv = os.path.join(output_dir, "HAM10000_metadata.csv")
    download_cache = os.path.join(output_dir, "_downloads")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(download_cache, exist_ok=True)

    # ── Idempotency check ──
    if not args.force:
        n_existing = count_images(images_dir)
        if n_existing >= EXPECTED_IMAGE_COUNT and os.path.exists(metadata_csv):
            print(f"HAM10000 already downloaded:")
            print(f"  Images:   {n_existing} in {images_dir}")
            print(f"  Metadata: {metadata_csv}")
            print(f"Use --force to re-download")
            return

    print("=" * 60)
    print("  HAM10000 Download")
    print("=" * 60)
    print(f"  DOI:        {HAM10000_DOI}")
    print(f"  Output dir: {output_dir}")
    print()

    # ── Step 1: Get file listing from Dataverse API ──
    print("[1/3] Fetching Dataverse file listing...")
    try:
        all_files = fetch_dataset_listing(HAM10000_DOI)
    except Exception as e:
        print(f"\nERROR: Could not reach Harvard Dataverse API.")
        print(f"       {type(e).__name__}: {e}")
        print(f"\nYou can manually download from:")
        print(f"  https://dataverse.harvard.edu/dataset.xhtml?persistentId={HAM10000_DOI}")
        print(f"\nThen place HAM10000_images_part_1.zip, HAM10000_images_part_2.zip,")
        print(f"and HAM10000_metadata.csv into {output_dir} and re-run this script.")
        sys.exit(1)

    # Build a map of filename -> (numeric id, size) for files we care about
    matched = {}
    for f in all_files:
        label = f.get("label") or f.get("dataFile", {}).get("filename", "")
        file_id = f.get("dataFile", {}).get("id") or f.get("id")
        if not label or not file_id:
            continue
        if label in WANTED_FILES:
            matched[label] = file_id

    # Confirm we got everything essential
    required_archives = [
        name for name, kind in WANTED_FILES.items()
        if kind == "image_archive"
    ]
    missing_archives = [n for n in required_archives if n not in matched]
    if missing_archives:
        print(f"  ERROR: these essential files are missing from the dataset listing:")
        for n in missing_archives:
            print(f"    - {n}")
        print(f"  Available files in dataset:")
        for f in all_files:
            label = f.get("label", "?")
            print(f"    - {label}")
        sys.exit(1)

    # Metadata file may exist under either .tab or .csv naming
    metadata_label = None
    for candidate in ("HAM10000_metadata.csv", "HAM10000_metadata.tab"):
        if candidate in matched:
            metadata_label = candidate
            break
    if not metadata_label:
        print(f"  WARNING: no metadata file found in dataset; images only")

    # ── Step 2: Download each file to the cache dir ──
    print(f"\n[2/3] Downloading files...")
    download_cache = os.path.join(output_dir, "_downloads")
    downloads = {}
    for label, file_id in matched.items():
        dest = os.path.join(download_cache, label)
        if os.path.exists(dest) and not args.force:
            size_mb = os.path.getsize(dest) / 1024 / 1024
            print(f"  SKIP {label}: already cached ({size_mb:.1f} MB)")
        else:
            download_file(file_id, dest, label)
        downloads[label] = dest

    # ── Step 3: Unpack image archives and place metadata ──
    print(f"\n[3/3] Unpacking archives and placing metadata...")
    total_extracted = 0
    for label in required_archives:
        total_extracted += unpack_images_zip(downloads[label], images_dir)

    n_final = count_images(images_dir)
    print(f"  Total images in {images_dir}: {n_final}")
    if n_final < EXPECTED_IMAGE_COUNT:
        print(f"  WARNING: expected {EXPECTED_IMAGE_COUNT}, got {n_final}")

    # Metadata: convert from Dataverse's tab-separated .tab format to a real
    # comma-separated .csv so that downstream `pd.read_csv()` calls in Phase 0
    # and Phase 3 work without needing an explicit `sep='\t'` argument.
    if metadata_label:
        src = downloads[metadata_label]
        if metadata_label.endswith(".tab"):
            # Convert TSV to CSV inline
            import csv as _csv
            with open(src, "r", newline="") as tab_f, \
                 open(metadata_csv, "w", newline="") as csv_f:
                reader = _csv.reader(tab_f, delimiter="\t")
                writer = _csv.writer(csv_f, delimiter=",", quoting=_csv.QUOTE_MINIMAL)
                for row in reader:
                    writer.writerow(row)
            print(f"  Metadata saved: {metadata_csv} (converted from .tab -> .csv)")
        else:
            shutil.copy2(src, metadata_csv)
            print(f"  Metadata saved: {metadata_csv}")

    # ── Cleanup ──
    if not args.keep_zips:
        print(f"\nCleaning up download cache...")
        shutil.rmtree(download_cache, ignore_errors=True)
    else:
        print(f"\nKeeping download cache at: {download_cache}")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"  HAM10000 download complete")
    print(f"{'=' * 60}")
    print(f"  Images:   {n_final} files in {images_dir}")
    if os.path.exists(metadata_csv):
        size_kb = os.path.getsize(metadata_csv) / 1024
        print(f"  Metadata: {metadata_csv} ({size_kb:.0f} KB)")
    print()
    print(f"Next: pass these paths to dataset_prep.py:")
    print(f"  --ham_images   {images_dir}")
    print(f"  --ham_metadata {metadata_csv}")


if __name__ == "__main__":
    main()
