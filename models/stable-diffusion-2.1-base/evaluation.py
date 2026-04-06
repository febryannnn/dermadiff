# Eval test set
EXP_C2_TEST_LOCAL = f'/content/exp_c2_{SELECTED_RATIO}_test'
os.makedirs(EXP_C2_TEST_LOCAL, exist_ok=True)
BEST_CKPT = os.path.join(EXP_C2_LOCAL, 'checkpoint-best.pth')

!cd {PANDERM_DIR}/classification && \
CUDA_VISIBLE_DEVICES=0 python3 run_class_finetuning.py \
    --model {CONFIG['model_name']} \
    --nb_classes {CONFIG['nb_classes']} \
    --batch_size {CONFIG['batch_size']} \
    --sin_pos_emb \
    --imagenet_default_mean_and_std \
    --eval \
    --resume {BEST_CKPT} \
    --csv_path {CSV_PATH} \
    --root_path "/" \
    --output_dir {EXP_C2_TEST_LOCAL}

print("Test evaluation done!")

# Classification Report
import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             confusion_matrix, roc_auc_score)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

test_df = pd.read_csv(os.path.join(EXP_C2_TEST_LOCAL, 'test.csv'))
y_true = test_df['true_label'].values
y_pred = test_df['predicted_label'].values
y_prob = test_df[[f'probability_class_{i}' for i in range(7)]].values
y_true_bin = label_binarize(y_true, classes=range(7))

class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
acc = accuracy_score(y_true, y_pred)
wf1 = f1_score(y_true, y_pred, average='weighted')
mf1 = f1_score(y_true, y_pred, average='macro')
macro_auc = roc_auc_score(y_true_bin, y_prob, average='macro', multi_class='ovr')

print("=" * 60)
print(f"EXPERIMENT C2 — TEST SET (SD 2.0, {SELECTED_RATIO})")
print("=" * 60)
print(f"\n  Overall accuracy:  {acc:.4f}")
print(f"  Weighted F1:       {wf1:.4f}")
print(f"  Macro F1:          {mf1:.4f}")
print(f"  AUC-ROC macro:     {macro_auc:.4f}")

prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=range(7))
print(f"\n  PER-CLASS METRICS:")
print(f"  {'Class':>8s}  {'Prec':>6s}  {'Recall':>6s}  {'F1':>6s}  {'Support':>7s}")
print(f"  {'-'*45}")
for i in range(7):
    print(f"  {class_names[i]:>8s}  {prec[i]:6.3f}  {rec[i]:6.3f}  {f1[i]:6.3f}  {sup[i]:7d}")

macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
print(f"\n  AVERAGE METRICS:")
print(f"  {'':>8s}  {'Prec':>6s}  {'Recall':>6s}  {'F1':>6s}")
print(f"  {'-'*35}")
print(f"  {'macro':>8s}  {macro_prec:6.3f}  {macro_rec:6.3f}  {macro_f1:6.3f}")
print(f"  {'weighted':>8s}  {weighted_prec:6.3f}  {weighted_rec:6.3f}  {weighted_f1:6.3f}")

cm = confusion_matrix(y_true, y_pred, labels=range(7))
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title(f'Confusion Matrix — Exp C2 TEST ({SELECTED_RATIO})')
plt.tight_layout()
plt.savefig(os.path.join(EXP_C2_TEST_LOCAL, 'confusion_matrix_test.png'), dpi=150)
plt.show()

risk_map = {'nv':0, 'bkl':0, 'df':0, 'vasc':0, 'akiec':1, 'mel':2, 'bcc':2}
label_to_risk = {i: risk_map[class_names[i]] for i in range(7)}
y_true_risk = np.array([label_to_risk[y] for y in y_true])
y_pred_risk = np.array([label_to_risk[y] for y in y_pred])
print(f"\n  3-LEVEL RISK ACCURACY: {accuracy_score(y_true_risk, y_pred_risk):.4f}")

# Save to Drive
import shutil

EXP_C2_DRIVE = os.path.join(NOTEBOOK_DIR, f'outputs/exp_c2_{SELECTED_RATIO}')
os.makedirs(EXP_C2_DRIVE, exist_ok=True)
for f in os.listdir(EXP_C2_TEST_LOCAL):
    shutil.copy2(os.path.join(EXP_C2_TEST_LOCAL, f), EXP_C2_DRIVE)

SHARED_RESULTS = os.path.join(SHARED, f'results/exp_c2_{SELECTED_RATIO}')
os.makedirs(SHARED_RESULTS, exist_ok=True)
shutil.copytree(EXP_C2_DRIVE, SHARED_RESULTS, dirs_exist_ok=True)
print("All results saved!")