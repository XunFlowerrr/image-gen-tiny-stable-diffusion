import os
import re
import torch
import matplotlib.pyplot as plt
from PIL import Image
from contextlib import nullcontext
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    UNet2DConditionModel
)

# ───────────────────────────────────────────────────────────────────────────────
# 0) Configuration
# ───────────────────────────────────────────────────────────────────────────────
CHECKPOINT_STEP   = 200000
CKPT_DIR          = f"./image-gen/model/unet_step_{CHECKPOINT_STEP:06d}"
BASE_DIR          = "./image-gen/model/mini_sd_final"
BASE_OUT_DIR      = "./image-gen/out/prompt_grids"
os.makedirs(BASE_OUT_DIR, exist_ok=True)

# ───────────────────────────────────────────────────────────────────────────────
# 0.1) Auto-incrementing subdirectory name
# ───────────────────────────────────────────────────────────────────────────────
base_name = f"prompt_grids_{CHECKPOINT_STEP}"
pattern   = re.compile(rf"^{re.escape(base_name)}_(\d+)$")
existing  = [
    int(m.group(1))
    for name in os.listdir(BASE_OUT_DIR)
    if (m := pattern.match(name))
]
next_seq = max(existing) + 1 if existing else 0
OUT_DIR  = os.path.join(BASE_OUT_DIR, f"{base_name}_{next_seq}")
os.makedirs(OUT_DIR, exist_ok=True)
print(f"💾 Output directory: {OUT_DIR}")

# ───────────────────────────────────────────────────────────────────────────────
# 0.2) Decide device & precision
# ───────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.float16 if device.type == "cuda" else torch.float32
print(f"🚀 Running on {device} with dtype={dtype}")

# ───────────────────────────────────────────────────────────────────────────────
# 1) Inference grid settings
# # ───────────────────────────────────────────────────────────────────────────────
# N_STEPS         = [5, 10, 15, 30, 50, 75]
# GUIDANCE_SCALES = [3.0, 5.0, 7.5, 10.0]
N_STEPS         = [50,75]
GUIDANCE_SCALES = [5.0, 7.5]
# PROMPTS = [
#     "A red apple on a wooden table",
#     "A black cat sitting in a sunlit window",
#     "A small boat on a calm lake at sunrise",
#     "A golden retriever playing in autumn leaves",
#     "Snow-capped mountain under a clear blue sky",
#     "A plate of sushi on a bamboo mat",
#     "A bright sunflower field in summer",
#     "A close-up of a water droplet on a green leaf",
#     "A vintage car parked in front of a brick building",
#     "A hot air balloon floating over rolling hills",
#     "A skateboarder doing a trick on a ramp",
#     "A steaming cup of coffee on a patio table",
#     "A group of colorful parrots in a jungle",
#     "A neon sign on a rainy city street at night",
#     "A kitten chasing a ball of yarn",
#     "A surfer riding a big ocean wave",
#     "A ballerina dancing on a stage under a spotlight",
#     "A glass of red wine on a marble countertop",
#     "Fireworks exploding over a river at night",
#     "A bouquet of roses in a clear glass vase",
#     "An astronaut floating in space with Earth below",
#     "A robot chef cooking pancakes in a kitchen",
#     "A whimsical treehouse high in a forest canopy",
#     "A dragon curled around a medieval castle tower",
#     "A neon-lit diner on a dark highway",
#     "A stack of books next to a vintage reading lamp",
#     "A lion roaring on the African savannah",
#     "A colorful hot rod car parked in the desert",
#     "A child flying a kite on a windy beach",
#     "A cartoon panda eating bamboo shoots",
# ]

PROMPTS= [
    "Noodle soup in a bowl with chopsticks",
    "A women tennis player"
]

# ───────────────────────────────────────────────────────────────────────────────
# 2) Load base pipeline (VAE, text encoder, tokenizer, scheduler config)
# ───────────────────────────────────────────────────────────────────────────────
print("🔧 Loading base pipeline …")
pipe_base = StableDiffusionPipeline.from_pretrained(
    BASE_DIR,
    torch_dtype=dtype,
    local_files_only=True
).to(device)

sched_config = pipe_base.scheduler.config

# ───────────────────────────────────────────────────────────────────────────────
# 3) Load UNet checkpoint
# ───────────────────────────────────────────────────────────────────────────────
print(f"⟳ Loading UNet @ step {CHECKPOINT_STEP} …")
unet = UNet2DConditionModel.from_pretrained(
    CKPT_DIR,
    torch_dtype=dtype,
    local_files_only=True
).to(device)

pipe = StableDiffusionPipeline(
    vae=pipe_base.vae,
    text_encoder=pipe_base.text_encoder,
    tokenizer=pipe_base.tokenizer,
    unet=unet,
    scheduler=DDIMScheduler.from_config(sched_config),
    safety_checker=None,
    feature_extractor=None
).to(device)

# ───────────────────────────────────────────────────────────────────────────────
# 4) Generate and save one grid per prompt
# ───────────────────────────────────────────────────────────────────────────────
for prompt in PROMPTS:
    slug     = re.sub(r'[^a-z0-9]+', '_', prompt.lower()).strip('_')
    out_path = os.path.join(OUT_DIR, f"{slug}.png")

    # build grid of images for this prompt
    grid = [[None for _ in GUIDANCE_SCALES] for _ in N_STEPS]
    for i, n_steps in enumerate(N_STEPS):
        for j, guidance in enumerate(GUIDANCE_SCALES):
            print(f"▶ Generating for prompt='{prompt}', steps={n_steps}, guidance={guidance}")

            # only use mixed-precision autocast on CUDA
            autocast_ctx = torch.autocast("cuda") if device.type == "cuda" else nullcontext()
            with autocast_ctx:
                images = pipe(
                    prompt,
                    num_inference_steps=n_steps,
                    guidance_scale=guidance
                ).images
            grid[i][j] = images[0]

    rows, cols = len(N_STEPS), len(GUIDANCE_SCALES)
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(5 * cols, 5 * rows),
        squeeze=False,
        # give extra room on the left for your y-labels:
        gridspec_kw={"left": 0.2}
    )
    for i in range(rows):
        for j in range(cols):
            ax = axes[i][j]
            ax.imshow(grid[i][j])

            # hide just the ticks & spines, but not the y-label:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            # column titles:
            if i == 0:
                ax.set_title(f"g={GUIDANCE_SCALES[j]}", fontsize=12)
            # row labels on first column:
            if j == 0:
                    ax.set_ylabel(
        f"steps={N_STEPS[i]}",
        fontsize=12,
        rotation=90,      # rotate 90°
        va="center",      # vertically center the label
        labelpad=10
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved grid for prompt '{prompt}' → {out_path}")
