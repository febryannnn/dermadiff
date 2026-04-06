# Clone PanDerm
if not os.path.exists('/content/PanDerm'):
    !git clone https://github.com/SiyuanYan1/PanDerm.git
    !pip install -r /content/PanDerm/classification/requirements.txt -q
    print("PanDerm cloned & installed")
else:
    print("PanDerm already exists")

# Patch torch.load
finetuning_file = '/content/PanDerm/classification/run_class_finetuning.py'
with open(finetuning_file, 'r') as f:
    code = f.read()
code = code.replace('torch.load(model_weight)', 'torch.load(model_weight, weights_only=False)')
code = code.replace('torch.load(args.resume', 'torch.load(args.resume, weights_only=False')
with open(finetuning_file, 'w') as f:
    f.write(code)
print(f"Patched {code.count('weights_only=False')} torch.load() calls")

import os
os.environ['WANDB_MODE'] = 'disabled'

SELECTED_RATIO = '5x'
# CSV_PATH = os.path.join(NOTEBOOK_DIR, f'temp/ham10000_exp_c2_{SELECTED_RATIO}.csv')
CSV_PATH = os.path.join(NOTEBOOK_DIR, f'temp/ham10000_exp_c2_{SELECTED_RATIO}_filtered_v2.csv')
PANDERM_DIR = '/content/PanDerm'

EXP_C2_LOCAL = f'/content/exp_c2_{SELECTED_RATIO}_output'
os.makedirs(EXP_C2_LOCAL, exist_ok=True)

exp_c2_cmd = f"""cd {PANDERM_DIR}/classification && \
CUDA_VISIBLE_DEVICES=0 python3 run_class_finetuning.py \
    --model {CONFIG['model_name']} \
    --pretrained_checkpoint {WEIGHT_FILE} \
    --nb_classes {CONFIG['nb_classes']} \
    --batch_size {CONFIG['batch_size']} \
    --lr {CONFIG['lr']} \
    --update_freq 1 \
    --warmup_epochs {CONFIG['warmup_epochs']} \
    --epochs {CONFIG['epochs']} \
    --layer_decay {CONFIG['layer_decay']} \
    --drop_path {CONFIG['drop_path']} \
    --weight_decay {CONFIG['weight_decay']} \
    --mixup {CONFIG['mixup']} \
    --cutmix {CONFIG['cutmix']} \
    --weights \
    --sin_pos_emb \
    --no_auto_resume \
    --imagenet_default_mean_and_std \
    --exp_name "DermaDiff_ExpC2_{SELECTED_RATIO}" \
    --output_dir {EXP_C2_LOCAL} \
    --csv_path {CSV_PATH} \
    --root_path "/" \
    --seed {CONFIG['training_seeds'][0]}"""

print("Starting Exp C2 training...")
!{exp_c2_cmd}

import shutil
EXP_C2_DRIVE = os.path.join(NOTEBOOK_DIR, f'outputs/exp_c2_{SELECTED_RATIO}')
os.makedirs(EXP_C2_DRIVE, exist_ok=True)
shutil.copytree(EXP_C2_LOCAL, EXP_C2_DRIVE, dirs_exist_ok=True)
print(f"Training done, copied to {EXP_C2_DRIVE}")