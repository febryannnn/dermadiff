import os
import subprocess
import argparse


# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune PanDerm classifier")

    # path config
    parser.add_argument("--project-root", type=str, default=".",
                        help="Project root folder (default: current directory)")
    parser.add_argument("--exp-name", type=str, default="exp_c2",
                        help="Experiment folder name under notebooks/")

    # PanDerm repo location (outside project-root by default since it's a 3rd-party dep)
    parser.add_argument("--panderm-dir", type=str, default="./PanDerm",
                        help="Path to PanDerm repo (will be cloned if missing)")
    parser.add_argument("--panderm-repo-url", type=str,
                        default="https://github.com/SiyuanYan1/PanDerm.git")

    # CSV selection
    parser.add_argument("--ratio", type=str, default="5x",
                        help="Synthetic data ratio (e.g. 1x, 5x)")
    parser.add_argument("--csv-suffix", type=str, default="_filtered_v2",
                        help="CSV filename suffix (e.g. '_filtered_v2' or '')")

    # exp name for logging
    parser.add_argument("--run-name", type=str, default="DermaDiff_ExpC2",
                        help="Run name for logging/checkpoints")

    return parser.parse_args()


# Clone & Patch PanDerm
def setup_panderm(panderm_dir, repo_url):
    if not os.path.exists(panderm_dir):
        print(f"Cloning PanDerm to {panderm_dir}...")
        subprocess.check_call(["git", "clone", repo_url, panderm_dir])

        req_file = os.path.join(panderm_dir, "classification/requirements.txt")
        if os.path.exists(req_file):
            print("Installing PanDerm requirements...")
            subprocess.check_call([
                "pip", "install", "-r", req_file, "-q"
            ])
        print("PanDerm cloned & installed")
    else:
        print(f"PanDerm already exists at {panderm_dir}")

    # patch torch.load for torch>=2.4 security default
    finetuning_file = os.path.join(panderm_dir, "classification/run_class_finetuning.py")
    with open(finetuning_file, "r") as f:
        code = f.read()

    patched = code
    patched = patched.replace(
        "torch.load(model_weight)",
        "torch.load(model_weight, weights_only=False)"
    )
    patched = patched.replace(
        "torch.load(args.resume",
        "torch.load(args.resume, weights_only=False"
    )

    if patched != code:
        with open(finetuning_file, "w") as f:
            f.write(patched)
        print(f"Patched torch.load() calls in {finetuning_file}")
    else:
        print("torch.load() already patched (or not found)")


# Build Training Command
def build_training_cmd(
    panderm_dir,
    config,
    weight_file,
    csv_path,
    output_dir,
    run_name,
    seed,
):
    # run from panderm_dir/classification
    cmd = [
        "python3", "run_class_finetuning.py",
        "--model", str(config["model_name"]),
        "--pretrained_checkpoint", str(weight_file),
        "--nb_classes", str(config["nb_classes"]),
        "--batch_size", str(config["batch_size"]),
        "--lr", str(config["lr"]),
        "--update_freq", "1",
        "--warmup_epochs", str(config["warmup_epochs"]),
        "--epochs", str(config["epochs"]),
        "--layer_decay", str(config["layer_decay"]),
        "--drop_path", str(config["drop_path"]),
        "--weight_decay", str(config["weight_decay"]),
        "--mixup", str(config["mixup"]),
        "--cutmix", str(config["cutmix"]),
        "--weights",
        "--sin_pos_emb",
        "--no_auto_resume",
        "--imagenet_default_mean_and_std",
        "--exp_name", run_name,
        "--output_dir", output_dir,
        "--csv_path", csv_path,
        "--root_path", "/",
        "--seed", str(seed),
    ]
    return cmd


# Main
def main():
    args = parse_args()

    # disable wandb
    os.environ["WANDB_MODE"] = "disabled"

    # resolve project paths
    project_root = os.path.abspath(args.project_root)
    shared_dir = os.path.join(project_root, "shared")
    notebook_dir = os.path.join(project_root, "notebooks", args.exp_name)
    panderm_dir = os.path.abspath(args.panderm_dir)

    # shared config
    import json
    with open(os.path.join(shared_dir, "config/shared_config.json")) as f:
        CONFIG = json.load(f)

    # pretrained weights path
    weight_file = os.path.join(shared_dir, "weights/panderm_ll_data6_checkpoint-499.pth")

    # CSV path (augmented with synthetic data)
    csv_path = os.path.join(
        notebook_dir,
        f"temp/ham10000_{args.exp_name}_{args.ratio}{args.csv_suffix}.csv"
    )

    # output dir for fine-tuned checkpoints
    output_dir = os.path.join(notebook_dir, f"temp/finetune_{args.ratio}_output")
    os.makedirs(output_dir, exist_ok=True)

    # final output copy (persistent location)
    final_output_dir = os.path.join(notebook_dir, f"outputs/{args.exp_name}_{args.ratio}")

    # sanity checks
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.exists(weight_file):
        raise FileNotFoundError(f"Pretrained weights not found: {weight_file}")

    # setup PanDerm
    setup_panderm(panderm_dir, args.panderm_repo_url)

    # build command
    run_name = f"{args.run_name}_{args.ratio}"
    cmd = build_training_cmd(
        panderm_dir=panderm_dir,
        config=CONFIG,
        weight_file=weight_file,
        csv_path=csv_path,
        output_dir=output_dir,
        run_name=run_name,
        seed=CONFIG["training_seeds"][0],
    )

    # run training from panderm classification dir
    working_dir = os.path.join(panderm_dir, "classification")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")

    print(f"\nStarting {args.exp_name} training ({args.ratio})...")
    print(f"  Working dir : {working_dir}")
    print(f"  CSV         : {csv_path}")
    print(f"  Output dir  : {output_dir}")
    print(f"  Run name    : {run_name}\n")

    subprocess.check_call(cmd, cwd=working_dir, env=env)

    # copy results to persistent outputs dir
    import shutil
    os.makedirs(final_output_dir, exist_ok=True)
    shutil.copytree(output_dir, final_output_dir, dirs_exist_ok=True)
    print(f"\nTraining done, copied to {final_output_dir}")


if __name__ == "__main__":
    main()