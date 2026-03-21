# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

import os
import pandas as pd
import shutil
from tqdm import tqdm

# Config
OUTPUT_DIR = "/content/drive/MyDrive/isic2019"
TEMP_DIR = "/content/isic2019_temp"  # Temp di local disk Colab (lebih cepat)

# Semua 8 kelas ISIC 2019
# Malignant:      MEL (4522), BCC (3323), SCC (628)
# Pre-malignant:  AK (867)
# Benign:         NV (12875), BKL (2624), DF (239), VASC (253)
ALL_CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Step 1: Download ZIP
print("=" * 60)
print("STEP 1: Download ISIC 2019 Training ZIP dari S3...")
print("=" * 60)
print("Ukuran file ±9GB, estimasi 5-15 menit\n")

ZIP_URL = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip"
GT_URL = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv"
META_URL = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Metadata.csv"

zip_path = os.path.join(TEMP_DIR, "ISIC_2019_Training_Input.zip")
gt_path = os.path.join(OUTPUT_DIR, "ISIC_2019_Training_GroundTruth.csv")
meta_path = os.path.join(OUTPUT_DIR, "ISIC_2019_Training_Metadata.csv")

# Download ZIP dengan wget (lebih cepat dan bisa resume)
if not os.path.exists(zip_path):
    print("Downloading ZIP file...")
    !wget -c -q --show-progress -O "{zip_path}" "{ZIP_URL}"
    print(f"\nZIP saved: {zip_path}")
else:
    size_gb = os.path.getsize(zip_path) / (1024**3)
    print(f"ZIP sudah ada: {zip_path} ({size_gb:.1f} GB)")

# Download ground truth
if not os.path.exists(gt_path):
    print("\nDownloading ground truth CSV...")
    !wget -q -O "{gt_path}" "{GT_URL}"
    print(f"Saved: {gt_path}")
else:
    print(f"Ground truth sudah ada: {gt_path}")

# Download metadata
if not os.path.exists(meta_path):
    print("Downloading metadata CSV...")
    !wget -q -O "{meta_path}" "{META_URL}"
    print(f"Saved: {meta_path}")
else:
    print(f"Metadata sudah ada: {meta_path}")

print("\nDownload selesai!")

# Step 2: Extract ZIP ke temp directory (di colab)
print("=" * 60)
print("STEP 2: Extracting ZIP...")
print("=" * 60)

extract_dir = os.path.join(TEMP_DIR, "ISIC_2019_Training_Input")

if not os.path.exists(extract_dir) or len(os.listdir(extract_dir)) < 25000:
    print("Extracting... (ini bisa beberapa menit)")
    !unzip -q -o "{zip_path}" -d "{TEMP_DIR}"

    # Cek apakah ada nested folder
    if os.path.exists(os.path.join(TEMP_DIR, "ISIC_2019_Training_Input")):
        extract_dir = os.path.join(TEMP_DIR, "ISIC_2019_Training_Input")

    num_files = len([f for f in os.listdir(extract_dir) if f.endswith('.jpg')])
    print(f"Extracted: {num_files} images")
else:
    num_files = len([f for f in os.listdir(extract_dir) if f.endswith('.jpg')])
    print(f"Sudah di-extract: {num_files} images")

# Step 3: Parsing ground truth & sort gambar ke folder per kelas
print("\n" + "=" * 60)
print("STEP 3: Sorting images ke folder per kelas...")
print("=" * 60)

# Load ground truth
df_gt = pd.read_csv(gt_path)
print(f"Total images dalam ground truth: {len(df_gt)}")

# Tentukan diagnosis per image
available_classes = [c for c in ALL_CLASSES if c in df_gt.columns]

def get_diagnosis(row):
    for cls in available_classes:
        if row[cls] == 1.0:
            return cls
    return "UNK"

df_gt["diagnosis"] = df_gt.apply(get_diagnosis, axis=1)

# Tampilkan distribusi
print(f"\nDistribusi per kelas:")
print("-" * 40)
class_counts = df_gt["diagnosis"].value_counts()
for cls in available_classes:
    count = class_counts.get(cls, 0)
    label = "malignant" if cls in ["MEL", "BCC", "SCC"] else "pre-malignant" if cls == "AK" else "benign"
    print(f"  {cls:>4}: {count:>6} images  ({label})")
if "UNK" in class_counts:
    print(f"  {'UNK':>4}: {class_counts['UNK']:>6} images  (unknown)")
print(f"  {'':>4}  ------")
print(f"  Total: {len(df_gt):>5} images")

# Simpan labels CSV
labels_path = os.path.join(OUTPUT_DIR, "labels_all.csv")
df_gt.to_csv(labels_path, index=False)
print(f"\nLabels saved: {labels_path}")

# Buat folder per kelas di Google Drive
all_folders = available_classes + (["UNK"] if "UNK" in df_gt["diagnosis"].values else [])
for cls in all_folders:
    os.makedirs(os.path.join(OUTPUT_DIR, "images", cls), exist_ok=True)

# Hitung yang sudah ada di Drive (untuk resume)
already_in_drive = set()
for cls in all_folders:
    cls_dir = os.path.join(OUTPUT_DIR, "images", cls)
    if os.path.exists(cls_dir):
        for f in os.listdir(cls_dir):
            if f.endswith('.jpg'):
                already_in_drive.add(f.replace(".jpg", ""))

print(f"\nSudah ada di Drive: {len(already_in_drive)}")
print(f"Perlu dicopy      : {len(df_gt) - len(already_in_drive)}")

# Copy gambar dari temp ke Drive per kelas
skipped = 0
copied = 0
not_found = 0

for _, row in tqdm(df_gt.iterrows(), total=len(df_gt), desc="Sorting images"):
    image_id = row["image"]
    diagnosis = row["diagnosis"]

    # Skip jika sudah ada di Drive
    if image_id in already_in_drive:
        skipped += 1
        continue

    src = os.path.join(extract_dir, f"{image_id}.jpg")
    dst = os.path.join(OUTPUT_DIR, "images", diagnosis, f"{image_id}.jpg")

    if os.path.exists(src):
        shutil.copy2(src, dst)
        copied += 1
    else:
        not_found += 1

print(f"\n{'=' * 60}")
print(f"SORTING SELESAI!")
print(f"{'=' * 60}")
print(f"Copied    : {copied}")
print(f"Skipped   : {skipped} (sudah ada)")
print(f"Not found : {not_found}")

# Step 4: Verifikasi dan Summmary
print("\n" + "=" * 60)
print("STEP 4: Verifikasi")
print("=" * 60)

total_downloaded = 0
for cls in all_folders:
    cls_dir = os.path.join(OUTPUT_DIR, "images", cls)
    if os.path.exists(cls_dir):
        count = len([f for f in os.listdir(cls_dir) if f.endswith('.jpg')])
        expected = len(df_gt[df_gt["diagnosis"] == cls])
        total_downloaded += count
        status = "OK" if count == expected else f"INCOMPLETE ({count}/{expected})"
        print(f"  {cls:>4}: {count:>6} / {expected:<6} [{status}]")

print(f"\n  Total  : {total_downloaded}")
print(f"  Expected: {len(df_gt)}")

print(f"\nStruktur folder:")
print(f"  {OUTPUT_DIR}/")
print(f"  ├── labels_all.csv")
print(f"  ├── ISIC_2019_Training_GroundTruth.csv")
print(f"  ├── ISIC_2019_Training_Metadata.csv")
print(f"  └── images/")
for i, cls in enumerate(all_folders):
    connector = "└──" if i == len(all_folders) - 1 else "├──"
    cls_dir = os.path.join(OUTPUT_DIR, "images", cls)
    count = len([f for f in os.listdir(cls_dir) if f.endswith('.jpg')]) if os.path.exists(cls_dir) else 0
    print(f"      {connector} {cls}/  ({count} images)")

print("\nDone! Dataset ISIC 2019 lengkap siap digunakan.")


import os

OUTPUT_DIR = "/content/drive/MyDrive/isic2019"

EXPECTED = {
    "MEL": 4522, "NV": 12875, "BCC": 3323, "AK": 867,
    "BKL": 2624, "DF": 239, "VASC": 253, "SCC": 628
}

print("=" * 55)
print("  VERIFIKASI ISIC 2019 DATASET")
print("=" * 55)

# Cek CSV files
for csv_file in ["ISIC_2019_Training_GroundTruth.csv", "ISIC_2019_Training_Metadata.csv", "labels_all.csv"]:
    path = os.path.join(OUTPUT_DIR, csv_file)
    status = "ada" if os.path.exists(path) else "TIDAK ADA"
    print(f"  {csv_file}: {status}")

print()
total_actual = 0
total_expected = 0
all_ok = True

print(f"  {'Kelas':<6} {'Actual':>7} {'Expected':>9}  {'Status'}")
print(f"  {'-'*6} {'-'*7} {'-'*9}  {'-'*12}")

for cls in ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]:
    cls_dir = os.path.join(OUTPUT_DIR, "images", cls)
    exp = EXPECTED[cls]
    total_expected += exp

    if os.path.exists(cls_dir):
        count = len([f for f in os.listdir(cls_dir) if f.lower().endswith('.jpg')])
        total_actual += count
        if count == exp:
            status = "OK"
        elif count > 0:
            status = f"kurang {exp - count}"
            all_ok = False
        else:
            status = "KOSONG"
            all_ok = False
    else:
        count = 0
        status = "FOLDER TIDAK ADA"
        all_ok = False

    print(f"  {cls:<6} {count:>7} {exp:>9}  {status}")

print(f"  {'-'*6} {'-'*7} {'-'*9}  {'-'*12}")
print(f"  {'TOTAL':<6} {total_actual:>7} {total_expected:>9}  ", end="")

if all_ok:
    print("LENGKAP!")
else:
    pct = (total_actual / total_expected * 100) if total_expected > 0 else 0
    print(f"{pct:.1f}% complete")

print("=" * 55)