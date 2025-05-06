import os
import torch
import matplotlib.pyplot as plt
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    UNet2DConditionModel
)

# ----------------------------------------------------
# 1) Load the “base” pipeline (VAE, text encoder, tokenizer)
# ----------------------------------------------------
base_dir = "./image-gen/model/mini_sd_final"
pipe_base = StableDiffusionPipeline.from_pretrained(
    base_dir,
    torch_dtype=torch.float16,
    local_files_only=True
).to("cuda")
sched_config = pipe_base.scheduler.config

# ----------------------------------------------------
# 2) Specify which UNet checkpoints to compare
# ----------------------------------------------------
checkpoint_steps = [40000,80000,120000,160000,200000]

# ----------------------------------------------------
# 3) Prompts to generate for comparison
# ----------------------------------------------------
prompts = [
    "a bird on a car",
    "a baseball field",
    "a tree on the mountain"
]

# ----------------------------------------------------
# 4) Load each checkpoint, build a fresh pipeline, and generate
# ----------------------------------------------------
results = {}
for step in checkpoint_steps:
    ckpt_dir = f"./image-gen/model/unet_step_{step:06d}"
    print(f"⟳ Loading UNet @ step {step} from {ckpt_dir} …")

    unet_ckpt = UNet2DConditionModel.from_pretrained(
        ckpt_dir,
        torch_dtype=torch.float16,
        local_files_only=True
    ).to("cuda")

    pipe = StableDiffusionPipeline(
        vae=pipe_base.vae,
        text_encoder=pipe_base.text_encoder,
        tokenizer=pipe_base.tokenizer,
        unet=unet_ckpt,
        scheduler=DDIMScheduler.from_config(sched_config),
        safety_checker=None,
        feature_extractor=None
    ).to("cuda", torch.float16)

    imgs = []
    for prompt in prompts:
        with torch.autocast("cuda"):
            img = pipe(
                prompt,
                num_inference_steps=32,
                guidance_scale=7.5
            ).images[0]
        imgs.append(img)

    results[step] = imgs

# ----------------------------------------------------
# 5) Plot side-by-side comparison and save
# ----------------------------------------------------
n_steps   = len(checkpoint_steps)
n_prompts = len(prompts)
fig, axes = plt.subplots(
    n_steps, n_prompts,
    figsize=(4 * n_prompts, 4 * n_steps),
    squeeze=False
)

for i, step in enumerate(checkpoint_steps):
    for j, img in enumerate(results[step]):
        ax = axes[i][j]
        ax.imshow(img)
        ax.axis("off")
        if i == 0:
            ax.set_title(prompts[j], fontsize=10, pad=6)
        if j == 0:
            ax.set_ylabel(f"step {step}", fontsize=12)

plt.tight_layout()

# --- Save to disk ---
out_dir = "./image-gen/out/evolution"
os.makedirs(out_dir, exist_ok=True)
save_path = os.path.join(out_dir, "evolution.png")
fig.savefig(save_path, bbox_inches="tight")
print(f"✅ Saved comparison grid to {save_path}")
