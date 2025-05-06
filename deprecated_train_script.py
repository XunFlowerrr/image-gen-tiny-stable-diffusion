ckpt_root = "./image-gen"

# ================================================================
# 📥  Load COCO dataset directly from Hugging Face Hub
# ================================================================

from datasets import load_dataset, DownloadConfig, Image as HFImage, Features, Value
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
import warnings, random

def load_hf_dataset(n:int=1000, cfg="2014", seed:int=0, streaming:bool=False):
    """Loads the COCO dataset from Hugging Face Hub."""
    ds = None
    try:
        print(f"⇢  Loading COCO/{cfg} dataset {'via streaming' if streaming else 'by downloading'}...")
        # Define features to ensure consistent structure, especially for captions
        if cfg == "2014":
            features = Features({
                'image': HFImage(),
                'image_id': Value(dtype='int64'),
                'caption_id': Value(dtype='int64'),
                'sentences': {'raw': Value(dtype='string'), 'tokens': [Value(dtype='string')], 'token_ids': [Value(dtype='int64')]}
            })
        else:
             features = Features({
                'image': HFImage(),
                'image_id': Value(dtype='int64'),
                'caption_id': Value(dtype='int64'),
                'sentences_raw': [Value(dtype='string')],
            })

        dl_cfg = DownloadConfig(resume_download=True, max_retries=10)
        ds = load_dataset("HuggingFaceM4/COCO", name=cfg, split="train",
                          trust_remote_code=True, streaming=streaming,
                          token=True, download_config=dl_cfg,
                          features=features)

        if not streaming:
            print("⇢  Shuffling downloaded dataset...")
            ds = ds.shuffle(seed=seed)
            if n < len(ds):
                 print(f"⇢  Selecting the first {n} examples after shuffling.")
                 ds = ds.select(range(n))
        else:
             print("⇢  Streaming dataset. Shuffling is limited. Selecting first n examples.")
             ds = ds.take(n)

        print("⇢  Casting image column...")
        ds = ds.cast_column("image", HFImage(decode=True))

        print(f"✅ Dataset loaded successfully.")
        return ds

    except Exception as e:
        warnings.warn(f"Dataset loading failed → {e}")
        return None

# 🔧 Load the dataset directly
# Set streaming=True if you have limited disk space or want faster startup (less ideal for shuffling)
# Set streaming=False to download first (requires disk space, better shuffling)
# Adjust 'n' if you want a smaller subset for faster testing
dataset_size = 100000
hf_dataset = load_hf_dataset(n=dataset_size, cfg="2014", seed=42, streaming=False)

if hf_dataset is None:
    raise RuntimeError("Error: Dataset preparation failed. Check logs above.")
else:
    try:
        print(f"Dataset ready with approximately {len(hf_dataset)} examples.")
    except TypeError:
        print(f"Dataset ready (streaming, length unknown until iteration).")


import torch
from torch.utils.data import Dataset

class HuggingFaceImageCaptionDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, preprocess_fn, max_length=77, dataset_cfg="2014"):
        self.ds = hf_dataset
        self.tokenizer = tokenizer
        self.preprocess_fn = preprocess_fn
        self.max_length = max_length
        self.dataset_cfg = dataset_cfg
        print(f"Initialized dataset wrapper with dataset of type: {type(self.ds)}")
        try:
            print(f"Underlying dataset length: {len(self.ds)}")
        except TypeError:
            print("Warning: Underlying dataset type does not support len().")


    def __len__(self):
        try:
            return len(self.ds)
        except TypeError:
            # Fallback if len() is not supported (shouldn't happen for non-streaming)
            print("Warning: Dataset does not have a defined length. Returning pre-defined dataset_size.")
            return dataset_size


    def __getitem__(self, idx):
        try:
            item = self.ds[idx]

            # Extract caption based on dataset config, with checks for key existence
            if self.dataset_cfg == "2014":
                caption = item.get('sentences', {}).get('raw', None)
                if caption is None:
                    raise ValueError(f"Missing 'sentences' or 'raw' key in item at index {idx} for 2014 config.")
            else:
                caption_list = item.get('sentences_raw', [])
                if not caption_list:
                     raise ValueError(f"Missing or empty 'sentences_raw' key in item at index {idx} for non-2014 config.")
                caption = caption_list[0]

            # Load and preprocess image, with checks
            img = item.get('image', None)
            if img is None:
                 raise ValueError(f"Missing 'image' key in item at index {idx}.")

            # Ensure image is a PIL Image (it should be with decode=True)
            if not isinstance(img, Image.Image):
                 raise TypeError(f"Item 'image' at index {idx} is not a PIL Image. Type: {type(img)}")

            if img.mode != "RGB":
                 img = img.convert("RGB")
            img_tensor = self.preprocess_fn(img)

            # Tokenize text
            tok_out = self.tokenizer(caption, padding="max_length",
                                     max_length=self.max_length, truncation=True, return_tensors="pt")
            input_ids = tok_out["input_ids"].squeeze()
            attention_mask = tok_out["attention_mask"].squeeze()

            return img_tensor, input_ids, attention_mask

        except IndexError as e:
             actual_len = -1
             try:
                 actual_len = len(self.ds)
             except TypeError:
                 pass
             raise IndexError(f"IndexError accessing item {idx}: {e}. Dataset actual length: {actual_len}.") from e
        except Exception as e:
            print(f"Error loading/processing item {idx}: {e}")
            raise RuntimeError(f"Failed to load or process item {idx} due to: {e}") from e

# 🔤  Text encoder + tokenizer (frozen)
# Ensure tok is defined *before* the Dataset class uses it
from transformers import CLIPTokenizer, CLIPTextModel

# import AutoencoderKL
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler

tok      = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_enc = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32",
                                         torch_dtype=torch.float16).to("cuda").eval()
for p in text_enc.parameters(): p.requires_grad_(False)

# 🎨  Pre-trained VAE (frozen)
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse",
                                    torch_dtype=torch.float16).to("cuda").eval()
for p in vae.parameters(): p.requires_grad_(False)

# 🎛️  Tiny U-Net (~75 M params)  —  256×256 → 32×32 latents (4 channels)
unet = UNet2DConditionModel(
    sample_size      = 32,
    in_channels      = 4,
    out_channels     = 4,
    cross_attention_dim = 512,
    block_out_channels   = (128, 256, 256, 512),
    down_block_types     = ("DownBlock2D","DownBlock2D","AttnDownBlock2D","AttnDownBlock2D"),
    up_block_types       = ("AttnUpBlock2D","AttnUpBlock2D","UpBlock2D","UpBlock2D"),
    layers_per_block     = 1,
    mid_block_type       = "UNetMidBlock2DCrossAttn",
    attention_head_dim   = 8
).to("cuda", torch.float16)

print("U-Net params:", sum(p.numel() for p in unet.parameters())/1e6, "M")

# 🕑 Noise scheduler
noise_sched = DDPMScheduler(num_train_timesteps=1000)

from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import time
import os

image_size = 256
preprocess = transforms.Compose([
    transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC), # Use updated enum
    transforms.CenterCrop(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),                 # [0,1]
    transforms.Normalize([0.5]*3,[0.5]*3)  # → [-1,1]
])

# Instantiate the Hugging Face dataset using the loaded hf_dataset object
hf_image_caption_dataset = HuggingFaceImageCaptionDataset(hf_dataset, tok, preprocess, dataset_cfg="2014")


batch_size = 8
# Set num_workers=0 if using streaming or encountering issues with multiprocessing
train_loader = DataLoader(hf_image_caption_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0,
                          pin_memory=True)

print(f"DataLoader created.")
try:
    print(f"DataLoader will iterate over approx {len(hf_image_caption_dataset)} examples.")
except TypeError:
     print(f"DataLoader created for a streaming dataset (length unknown).")


# ================================================================
# 🚂 Training loop – fp16 VAE + fp32 UNet, ready for Accelerate≥0.23
# ================================================================
from accelerate import Accelerator
from torch.cuda.amp import autocast
from tqdm.auto import tqdm
import torch, os
from transformers import get_cosine_schedule_with_warmup

grad_accum      = 2
lr              = 1e-4
max_train_steps = 75_000
save_every      = 25000
vae_scaling     = 0.18215
warmup_steps    = 1_000

# -- UNet stays in **fp32**  -------------------------------------------------
unet = UNet2DConditionModel(
    sample_size=32,
    in_channels=4,
    out_channels=4,
    cross_attention_dim=512,
    block_out_channels=(128,256,256,512),
    layers_per_block=1,
    attention_head_dim=8
).to("cuda")

optimizer = torch.optim.AdamW(unet.parameters(), lr=lr, weight_decay=1e-2)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=max_train_steps
)

# -- Accelerate prepares everything -----------------------------------------
accelerator = Accelerator(mixed_precision="fp16",
                          gradient_accumulation_steps=grad_accum)
unet, optimizer, train_loader, scheduler = accelerator.prepare(unet,
                                                               optimizer,
                                                               train_loader,
                                                               scheduler)

# Dtype helpers
weight_dtype = next(unet.parameters()).dtype       # fp32
vae_dtype    = next(vae.parameters()).dtype        # fp16
print("UNet ⇒", weight_dtype, " |  VAE ⇒", vae_dtype)

unet.train()
global_step = 0
os.makedirs(ckpt_root, exist_ok=True)

print("Starting training loop …")
for epoch in range(999):                        # break via max_train_steps
    if global_step >= max_train_steps: break
    progress = tqdm(train_loader,
                    desc=f"Epoch {epoch}",
                    disable=not accelerator.is_local_main_process)

    for imgs, ids, masks in progress:
        imgs = imgs.to("cuda", dtype=vae_dtype)

        with accelerator.accumulate(unet):
            # -- VAE encode (frozen / fp16) -------------------------------
            with torch.no_grad():
                latents = vae.encode(imgs).latent_dist.sample()
            latents = latents * vae_scaling                        # fp16
            latents = latents.to(weight_dtype)                     # cast → fp32

            # -- Add noise ------------------------------------------------
            bsz = latents.size(0)
            t   = torch.randint(0,
                                noise_sched.config.num_train_timesteps,
                                (bsz,),
                                device=latents.device).long()
            noise  = torch.randn_like(latents)
            noisy  = noise_sched.add_noise(latents, noise, t)

            # -- Text embeddings (frozen / fp16 → cast) ------------------
            with torch.no_grad():
                txt_emb = text_enc(ids,
                                    attention_mask=masks).last_hidden_state
            txt_emb = txt_emb.to(weight_dtype)

            # -- Forward UNet (fp32 params) ------------------------------
            preds = unet(noisy, t, encoder_hidden_states=txt_emb).sample
            loss  = torch.nn.functional.mse_loss(preds.float(), noise.float())

            # -- Back‑prop / opt step ------------------------------------
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if accelerator.is_main_process:
                    progress.set_postfix(loss=f"{loss.item():.4f}",
                                         lr=f"{scheduler.get_last_lr()[0]:.6f}",
                                         step=global_step)
                    if (global_step % save_every == 0 or
                        global_step == max_train_steps):
                        ckpt_dir = f"{ckpt_root}/model/unet_step_{global_step:06d}"
                        accelerator.unwrap_model(unet).save_pretrained(
                            ckpt_dir, safe_serialization=True)
                        print(f"💾  checkpoint saved → {ckpt_dir}")

        if global_step >= max_train_steps:
            break

print("✅  Training finished at step", global_step)

# ================================================================
# 💾  Build & save Stable‑Diffusion pipeline (checker OFF)
# ================================================================
from diffusers import StableDiffusionPipeline
import torch, os, gc

device, pipe_dtype = "cuda", torch.float16
gc.collect(); torch.cuda.empty_cache()

print("🔧  Creating StableDiffusionPipeline …")
pipe = StableDiffusionPipeline(
    vae             = vae,                                   # fp16, frozen
    text_encoder    = text_enc,                              # fp16, frozen
    tokenizer       = tok,
    unet            = accelerator.unwrap_model(unet),        # fp32 → fp16 on .to()
    scheduler       = noise_sched,
    safety_checker  = None,
    feature_extractor = None
).to(device, pipe_dtype)

save_dir = f"{ckpt_root}/model/mini_sd_final"
print(f"💾  Saving full pipeline to → {save_dir}")
os.makedirs(save_dir, exist_ok=True)
pipe.save_pretrained(save_dir, safe_serialization=True)
print("✅  Pipeline saved successfully.")

from diffusers import DDIMScheduler
import matplotlib.pyplot as plt
import itertools
import os
import torch
from tqdm.auto import tqdm

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.to(device)

output_dir = ckpt_root + "/out/images"
os.makedirs(output_dir, exist_ok=True)
print(f"Generated images will be saved to: {output_dir}")

# --- Prompt Matrix Definition ---
subjects = ["a small red cabin", "a futuristic robot", "a fluffy cat"]
actions_or_styles = ["sleeping peacefully", "exploring", "painted in the style of Van Gogh"]
settings = ["in a snowy forest at sunset", "on a neon-lit city street", "under a starry night sky"]

# --- Generation Parameters ---
num_inference_steps = 175
guidance_scale = 7.5

# --- Generate and Display Matrix ---
# Calculate grid size based on the number of combinations we want to show
# Example: Show Subject vs Setting, keeping Action/Style constant for this grid
fixed_action_style = actions_or_styles[0] # Choose one action/style for this grid
rows = len(subjects)
cols = len(settings)

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows + 1)) # Adjust figsize
fig.suptitle(f"Prompt Matrix: Subject vs Setting (Action/Style: '{fixed_action_style}')", fontsize=16, y=1.03)

print(f"Generating prompt matrix ({rows}x{cols}) with fixed action/style: '{fixed_action_style}'...")

for r_idx, subject in enumerate(subjects):
    for c_idx, setting in enumerate(settings):
        prompt = f"{subject} {fixed_action_style} {setting}"
        print(f"  Generating ({r_idx},{c_idx}): {prompt}")

        with torch.cuda.amp.autocast():
            image = pipe(prompt,
                         num_inference_steps=num_inference_steps,
                         guidance_scale=guidance_scale).images[0]

        img_filename = f"matrix_subj{r_idx}_sett{c_idx}_{fixed_action_style.replace(' ', '_')[:15]}.png"
        img_path = os.path.join(output_dir, img_filename)
        image.save(img_path)
        print(f"    Saved image to: {img_path}")

        # Display image in the grid
        ax = axes[r_idx, c_idx] if rows > 1 and cols > 1 else (axes[max(r_idx, c_idx)] if rows > 1 or cols > 1 else axes)
        ax.imshow(image)
        ax.axis("off")

        if c_idx == 0:
            ax.text(-0.1, 0.5, subject, transform=ax.transAxes, rotation=90,
                    verticalalignment='center', horizontalalignment='right', fontsize=10)
        if r_idx == 0:
             ax.set_title(setting, fontsize=10, pad=10)

# Adjust layout
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()

# Save the matrix plot
matrix_plot_filename = f"prompt_matrix_{fixed_action_style.replace(' ', '_')[:15]}_plot.png"
matrix_plot_path = os.path.join(output_dir, matrix_plot_filename)
fig.savefig(matrix_plot_path, bbox_inches='tight')
print(f"Saved matrix plot to: {matrix_plot_path}")

# --- Optional: Generate for all combinations (can take a long time) ---
generate_all = False # Set to True to generate all combinations

if generate_all:
    all_prompts = [" ".join(combo) for combo in itertools.product(subjects, actions_or_styles, settings)]
    print(f"\nGenerating all {len(all_prompts)} combinations (this might take a while)...")

    # Determine grid size for all images (example: 5 columns)
    num_all_images = len(all_prompts)
    all_cols = 5
    all_rows = (num_all_images + all_cols - 1) // all_cols
    fig_all, axes_all = plt.subplots(all_rows, all_cols, figsize=(5 * all_cols, 5 * all_rows))
    axes_all = axes_all.flatten()

    for i, prompt in enumerate(tqdm(all_prompts, desc="Generating all combinations")):
        with torch.cuda.amp.autocast():
            image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]

        # Save the image to Google Drive
        img_filename = f"all_combo_{i:03d}.png"
        img_path = os.path.join(output_dir, img_filename)
        image.save(img_path)

        if i < len(axes_all):
            axes_all[i].imshow(image)
            axes_all[i].set_title(prompt, fontsize=6)
            axes_all[i].axis("off")

    # Hide unused subplots
    for j in range(i + 1, len(axes_all)):
        axes_all[j].axis("off")

    plt.tight_layout()
    plt.show()

    # Save the 'all combinations' plot
    all_plot_path = os.path.join(output_dir, "all_combinations_plot.png")
    fig_all.savefig(all_plot_path, bbox_inches='tight')
    print(f"Saved all combinations plot to: {all_plot_path}")

# --- Generate for a specific list of prompts ---
print("\nGenerating images for a specific list of prompts...")
specific_prompts = [
    "a small red cabin in a snowy forest at sunset, photorealistic",
    "a futuristic cityscape with flying cars, anime style",
    "a cute Corgi wearing a wizard hat, oil painting",
    "impressionist painting of sunflowers in a vase by Monet"
]
num_specific = len(specific_prompts)
cols_specific = 2 # Adjust grid columns as needed
rows_specific = (num_specific + cols_specific - 1) // cols_specific

fig_specific, axes_specific = plt.subplots(rows_specific, cols_specific, figsize=(5 * cols_specific, 5 * rows_specific))
axes_specific = axes_specific.flatten()

for i, prompt in enumerate(specific_prompts):
    print(f"  Generating: {prompt}")
    with torch.cuda.amp.autocast():
        image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

    # Save the image to Google Drive
    img_filename = f"specific_{i:03d}.png"
    img_path = os.path.join(output_dir, img_filename)
    image.save(img_path)
    print(f"    Saved image to: {img_path}")

    if i < len(axes_specific):
        axes_specific[i].imshow(image)
        axes_specific[i].set_title(prompt, fontsize=8)
        axes_specific[i].axis("off")

# Hide any unused subplots
for j in range(i + 1, len(axes_specific)):
    axes_specific[j].axis("off")

plt.tight_layout()
plt.show()

# Save the specific prompts plot
specific_plot_path = os.path.join(output_dir, "specific_prompts_plot.png")
fig_specific.savefig(specific_plot_path, bbox_inches='tight')
print(f"Saved specific prompts plot to: {specific_plot_path}")
