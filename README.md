# Small Stable Diffusion Model Training on COCO

This project focuses on training a relatively small Stable Diffusion U-Net model (~75M parameters) from scratch using the COCO dataset (specifically, the 2014 captions split).

## Key Components:

*   **`train_script.py`**: The main script for training the U-Net. It leverages pre-trained VAE and CLIP Text Encoder components, uses `accelerate` for mixed-precision training, saves U-Net checkpoints periodically, and finally saves the complete Stable Diffusion pipeline. It also includes example code for generating images post-training.
*   **`gen_checkpoint_evolution.py`**: A utility script to load different U-Net checkpoints saved during training and generate images for a fixed set of prompts. This helps visualize the model's learning progress over time.
*   **`gen_prompt_grid.py`**: Generates a grid of images based on systematically varying components of input prompts (e.g., subject, action, setting) using a specific trained checkpoint.
*   **`prep/`**: Contains scripts for data preparation and analysis:
    *   `k-mean.py`: Performs K-Means clustering on COCO captions using sentence embeddings.
    *   `zero.py`: Uses zero-shot classification to categorize COCO captions.

## Setup & Training:

1.  Ensure necessary libraries (`diffusers`, `transformers`, `datasets`, `accelerate`, `torch`, etc.) are installed.
2.  Configure paths and training parameters (like `dataset_size`, `max_train_steps`, `save_every`) in `train_script.py`.
3.  Run `train_script.py` to start the training process. Checkpoints and the final model will be saved under the `image-gen/model/` directory (or as configured by `ckpt_root`).
4.  Use the generation scripts (`gen_checkpoint_evolution.py`, `gen_prompt_grid.py`) or the generation section in `train_script.py` to create images with the trained model.
```
