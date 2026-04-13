import os
import json
import shutil
import subprocess
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score,
)
from sklearn.preprocessing import label_binarize


# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PanDerm on test set")

    # path config
    parser.add_argument("--project-root", type=str, default=".",
                        help="Project root folder (default: current directory)")
    parser.add_argument("--exp-name", type=str, default="exp_c2",
                        help="Experiment folder name under notebooks/")
    parser.add_argument("--panderm-dir", type=str, default="./PanDerm",
                        help="Path to PanDerm repo")

    # CSV selection
    parser.add_argument("--ratio", type=str, default="5x")
    parser.add_argument("--csv-suffix", type=str, default="_filtered_v2")

    # checkpoint (default: checkpoint-best.pth in finetune output dir)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (default: auto-detect best checkpoint)")

    return parser.parse_args()


# Class Names (7-class HAM10000)
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# 3-level risk mapping
RISK_MAP = {
    "nv": 0, "bkl": 0, "df": 0, "vasc": 0,
    "akiec": 1,
    "mel": 2, "bcc": 2,
}


# Build Eval Command
def build_eval_cmd(config, checkpoint, csv_path, output_dir):
    cmd = [
        "python3", "run_class_finetuning.py",
        "--model", str(config["model_name"]),
        "--nb_classes", str(config["nb_classes"]),
        "--batch_size", str(config["batch_size"]),
        "--sin_pos_emb",
        "--imagenet_default_mean_and_std",
        "--eval",
        "--resume", checkpoint,
        "--csv_path", csv_path,
        "--root_path", "/",
        "--output_dir", output_dir,
    ]
    return cmd


# Compute & Print Metrics
def compute_metrics(test_df, nb_classes, ratio_name, output_dir):
    y_true = test_df["true_label"].values
    y_pred = test_df["predicted_label"].values
    y_prob = test_df[[f"probability_class_{i}" for i in range(nb_classes)]].values
    y_true_bin = label_binarize(y_true, classes=range(nb_classes))

    acc = accuracy_score(y_true, y_pred)
    wf1 = f1_score(y_true, y_pred, average="weighted")
    mf1 = f1_score(y_true, y_pred, average="macro")
    macro_auc = roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")

    print("=" * 60)
    print(f"TEST SET RESULTS ({ratio_name})")
    print("=" * 60)
    print(f"\n  Overall accuracy:  {acc:.4f}")
    print(f"  Weighted F1:       {wf1:.4f}")
    print(f"  Macro F1:          {mf1:.4f}")
    print(f"  AUC-ROC macro:     {macro_auc:.4f}")

    # per-class metrics
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=range(nb_classes)
    )
    print(f"\n  PER-CLASS METRICS:")
    print(f"  {'Class':>8s}  {'Prec':>6s}  {'Recall':>6s}  {'F1':>6s}  {'Support':>7s}")
    print(f"  {'-'*45}")
    for i in range(nb_classes):
        print(f"  {CLASS_NAMES[i]:>8s}  {prec[i]:6.3f}  {rec[i]:6.3f}  {f1[i]:6.3f}  {sup[i]:7d}")

    # average metrics
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )
    print(f"\n  Average Metrics:")
    print(f"  {'':>8s}  {'Prec':>6s}  {'Recall':>6s}  {'F1':>6s}")
    print(f"  {'-'*35}")
    print(f"  {'macro':>8s}  {macro_prec:6.3f}  {macro_rec:6.3f}  {macro_f1:6.3f}")
    print(f"  {'weighted':>8s}  {weighted_prec:6.3f}  {weighted_rec:6.3f}  {weighted_f1:6.3f}")

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(nb_classes))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix - TEST ({ratio_name})")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix_test.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\n  Confusion matrix saved: {cm_path}")

    # 3-level risk accuracy
    label_to_risk = {i: RISK_MAP[CLASS_NAMES[i]] for i in range(nb_classes)}
    y_true_risk = np.array([label_to_risk[y] for y in y_true])
    y_pred_risk = np.array([label_to_risk[y] for y in y_pred])
    risk_acc = accuracy_score(y_true_risk, y_pred_risk)
    print(f"\n  3-LEVEL RISK ACCURACY: {risk_acc:.4f}")

    return {
        "accuracy": acc,
        "weighted_f1": wf1,
        "macro_f1": mf1,
        "macro_auc": macro_auc,
        "risk_accuracy": risk_acc,
    }


# Main
def main():
    args = parse_args()

    # resolve paths
    project_root = os.path.abspath(args.project_root)
    shared_dir = os.path.join(project_root, "shared")
    notebook_dir = os.path.join(project_root, "notebooks", args.exp_name)
    panderm_dir = os.path.abspath(args.panderm_dir)

    # load shared config
    with open(os.path.join(shared_dir, "config/shared_config.json")) as f:
        CONFIG = json.load(f)

    # CSV path (same as training)
    csv_path = os.path.join(
        notebook_dir,
        f"temp/ham10000_{args.exp_name}_{args.ratio}{args.csv_suffix}.csv"
    )

    # checkpoint path (auto-detect from finetune output)
    finetune_output = os.path.join(notebook_dir, f"temp/finetune_{args.ratio}_output")
    if args.checkpoint:
        checkpoint = args.checkpoint
    else:
        checkpoint = os.path.join(finetune_output, "checkpoint-best.pth")

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # eval output dir
    eval_output_dir = os.path.join(notebook_dir, f"temp/eval_{args.ratio}_test")
    os.makedirs(eval_output_dir, exist_ok=True)

    # run evaluation
    cmd = build_eval_cmd(
        config=CONFIG,
        checkpoint=checkpoint,
        csv_path=csv_path,
        output_dir=eval_output_dir,
    )

    working_dir = os.path.join(panderm_dir, "classification")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")

    print(f"Starting evaluation ({args.ratio})...")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  CSV       : {csv_path}\n")

    subprocess.check_call(cmd, cwd=working_dir, env=env)
    print("\nTest evaluation done!")

    # load predictions
    test_csv = os.path.join(eval_output_dir, "test.csv")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Prediction CSV not found: {test_csv}")

    test_df = pd.read_csv(test_csv)

    # compute & print classification report
    metrics = compute_metrics(
        test_df, CONFIG["nb_classes"], args.ratio, eval_output_dir
    )

    # save metrics summary as JSON
    metrics_path = os.path.join(eval_output_dir, "metrics_summary.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics summary saved: {metrics_path}")

    # copy all eval artifacts to persistent outputs dir
    final_output_dir = os.path.join(notebook_dir, f"outputs/{args.exp_name}_{args.ratio}")
    os.makedirs(final_output_dir, exist_ok=True)
    for fname in os.listdir(eval_output_dir):
        shutil.copy2(os.path.join(eval_output_dir, fname), final_output_dir)

    # copy to shared results dir (for comparing across experiments)
    shared_results = os.path.join(shared_dir, f"results/{args.exp_name}_{args.ratio}")
    os.makedirs(shared_results, exist_ok=True)
    shutil.copytree(final_output_dir, shared_results, dirs_exist_ok=True)

    print(f"\nAll results saved!")
    print(f"  Local  : {final_output_dir}")
    print(f"  Shared : {shared_results}")


if __name__ == "__main__":
    main()