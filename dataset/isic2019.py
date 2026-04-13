import os
import pandas as pd
import shutil
from tqdm import tqdm

import os

root_dir = "/Users/farelfebryan/kcvanguard/dermadiff-github"
output_dir = os.path.join(root_dir, "data", "isic2019")
temp_dir = os.path.join(root_dir, "data", "isic2019_temp")

# 7 Skin Lesion Class
ALL_CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC"]

os.makedirs(output_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)

# Step 1: Download ZIP
print("STEP 1: Download ISIC 2019 Training ZIP...")
print("-" * 60)
print("filesize ±9GB, 5-15 minutes estimation\n")

zip_url = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip"
ground_truth_url = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv"
META_URL = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Metadata.csv"

zip_path = os.path.join(temp_dir, "ISIC_2019_Training_Input.zip")
gt_path = os.path.join(output_dir, "ISIC_2019_Training_GroundTruth.csv")
meta_path = os.path.join(output_dir, "ISIC_2019_Training_Metadata.csv")

# using wget to download
if not os.path.exists(zip_path):
    print("Downloading ZIP file...")
    !wget -c -q --show-progress -O "{zip_path}" "{zip_url}"
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

# Step 2: Extract ZIP to temp directory
print("STEP 2: Extracting ZIP...")
print("-" * 60)

extract_dir = os.path.join(temp_dir, "ISIC_2019_Training_Input")

if not os.path.exists(extract_dir) or len(os.listdir(extract_dir)) < 25000:
    print("Extracting...")
    !unzip -q -o "{zip_path}" -d "{temp_dir}"

    # Cek apakah ada nested folder
    if os.path.exists(os.path.join(temp_dir, "ISIC_2019_Training_Input")):
        extract_dir = os.path.join(temp_dir, "ISIC_2019_Training_Input")

    num_files = len([f for f in os.listdir(extract_dir) if f.endswith('.jpg')])
    print(f"Extracted: {num_files} images")
else:
    num_files = len([f for f in os.listdir(extract_dir) if f.endswith('.jpg')])
    print(f"Sudah di-extract: {num_files} images")

# Step 3: Parsing ground truth & sort image to folder per class
print("\n")
print("STEP 3: Sorting images ke folder per kelas...")
print("-" * 60)

# Load ground truth
df_gt = pd.read_csv(gt_path)
print(f"Total images in ground truth: {len(df_gt)}")

available_classes = [c for c in ALL_CLASSES if c in df_gt.columns]

def get_diagnosis(row):
    for cls in available_classes:
        if row[cls] == 1.0:
            return cls
    return "UNK"

df_gt["diagnosis"] = df_gt.apply(get_diagnosis, axis=1)

# show distribution
print(f"\nPer-Class Distribution:")
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

# save csv labels
labels_path = os.path.join(output_dir, "labels_all.csv")
df_gt.to_csv(labels_path, index=False)
print(f"\nLabels saved: {labels_path}")

# make folder per class
all_folders = available_classes + (["UNK"] if "UNK" in df_gt["diagnosis"].values else [])
for cls in all_folders:
    os.makedirs(os.path.join(output_dir, "images", cls), exist_ok=True)

already_sorted = set()
for cls in all_folders:
    cls_dir = os.path.join(output_dir, "images", cls)
    if os.path.exists(cls_dir):
        for f in os.listdir(cls_dir):
            if f.endswith('.jpg'):
                already_sorted.add(f.replace(".jpg", ""))

print(f"\nAlreadt sort: {len(already_sorted)}")
print(f"Need to sort : {len(df_gt) - len(already_sorted)}")

skipped = 0
moved = 0
not_found = 0

for _, row in tqdm(df_gt.iterrows(), total=len(df_gt), desc="Sorting images"):
    image_id = row["image"]
    diagnosis = row["diagnosis"]

    if image_id in already_sorted:
        skipped += 1
        continue

    src = os.path.join(extract_dir, f"{image_id}.jpg")
    dst = os.path.join(output_dir, "images", diagnosis, f"{image_id}.jpg")

    if os.path.exists(src):
        shutil.move(src, dst)
        moved += 1
    else:
        not_found += 1

print(f"\n")
print(f"Finish Sorting!")
print(f"{'-' * 60}")
print(f"Moved     : {moved}")
print(f"Skipped   : {skipped}")
print(f"Not found : {not_found}")

# Optional: hapus temp directory untuk hemat disk
print("\nCleaning up temp files...")
try:
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
        print(f"  Removed: {extract_dir}")
    # Hapus ZIP juga kalau mau hemat disk (comment jika mau simpan buat backup)
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print(f"  Removed: {zip_path}")
except Exception as e:
    print(f"  Warning cleanup: {e}")

# Step 4: Verifikasi and Summary
print("\n")
print("STEP 4: Verifikasi")
print("-" * 60)

total_sorted = 0
for cls in all_folders:
    cls_dir = os.path.join(output_dir, "images", cls)
    if os.path.exists(cls_dir):
        count = len([f for f in os.listdir(cls_dir) if f.endswith('.jpg')])
        expected = len(df_gt[df_gt["diagnosis"] == cls])
        total_sorted += count
        status = "OK" if count == expected else f"INCOMPLETE ({count}/{expected})"
        print(f"  {cls:>4}: {count:>6} / {expected:<6} [{status}]")

print(f"\n  Total   : {total_sorted}")
print(f"  Expected: {len(df_gt)}")

print(f"\nFolder Structure:")
print(f"  {output_dir}/")
print(f"  ├── labels_all.csv")
print(f"  ├── ISIC_2019_Training_GroundTruth.csv")
print(f"  ├── ISIC_2019_Training_Metadata.csv")
print(f"  └── images/")
for i, cls in enumerate(all_folders):
    connector = "└──" if i == len(all_folders) - 1 else "├──"
    cls_dir = os.path.join(output_dir, "images", cls)
    count = len([f for f in os.listdir(cls_dir) if f.endswith('.jpg')]) if os.path.exists(cls_dir) else 0
    print(f"      {connector} {cls}/  ({count} images)")

print("\nDone! Dataset ISIC 2019 is ready!.")
print(f"Location: {output_dir}")


# Verifikasi
import os

output_dir = os.path.join(root_dir, "data", "isic2019")

EXPECTED = {
    "MEL": 4522, "NV": 12875, "BCC": 3323, "AK": 867,
    "BKL": 2624, "DF": 239, "VASC": 253, "SCC": 628
}

print("ISIC 2019 Dataset Verification")
print("-" * 55)

# Cek CSV files
for csv_file in ["ISIC_2019_Training_GroundTruth.csv", "ISIC_2019_Training_Metadata.csv", "labels_all.csv"]:
    path = os.path.join(output_dir, csv_file)
    status = "ada" if os.path.exists(path) else "TIDAK ADA"
    print(f"  {csv_file}: {status}")

print()
total_actual = 0
total_expected = 0
all_ok = True

print(f"  {'Class':<6} {'Actual':>7} {'Expected':>9}  {'Status'}")
print(f"  {'-'*6} {'-'*7} {'-'*9}  {'-'*12}")

for cls in ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]:
    cls_dir = os.path.join(output_dir, "images", cls)
    exp = EXPECTED[cls]
    total_expected += exp

    if os.path.exists(cls_dir):
        count = len([f for f in os.listdir(cls_dir) if f.lower().endswith('.jpg')])
        total_actual += count
        if count == exp:
            status = "OK"
        elif count > 0:
            status = f"Missed {exp - count}"
            all_ok = False
        else:
            status = "Empty!"
            all_ok = False
    else:
        count = 0
        status = "Folder Not Found!"
        all_ok = False

    print(f"  {cls:<6} {count:>7} {exp:>9}  {status}")

print(f"  {'-'*6} {'-'*7} {'-'*9}  {'-'*12}")
print(f"  {'Total':<6} {total_actual:>7} {total_expected:>9}  ", end="")

if all_ok:
    print("Complete!")
else:
    pct = (total_actual / total_expected * 100) if total_expected > 0 else 0
    print(f"{pct:.1f}% complete")

print("=" * 55)