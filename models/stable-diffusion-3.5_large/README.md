# Exp D - SD 3.5 Large LoRA (rank 64)

SD 3.5 Large + LoRA fine-tuning for minority-class skin lesion generation.

**Published result:** Macro F1 = 0.8482

## LoRA Weights (external download)

The SD 3.5 LoRA weights are ~180 MB each, which exceeds GitHub's 100 MB file
size limit. They are hosted on Google Drive instead.

**Download link:** [Google Drive - SD 3.5 LoRA Weights](https://drive.google.com/drive/folders/1TL7ucKRLBIMhyEKHX0JcZJT4m5DbWcrl?usp=sharing)

After downloading, place the weight files in the following structure:

```
models/stable-diffusion-3.5_large/LoRA Weights/
├── mel/pytorch_lora_weights.safetensors
├── bcc/pytorch_lora_weights.safetensors
├── akiec/pytorch_lora_weights.safetensors
├── df/pytorch_lora_weights.safetensors
└── vasc/pytorch_lora_weights.safetensors
```

You can verify the files are in place by checking that each is approximately
180 MB.
