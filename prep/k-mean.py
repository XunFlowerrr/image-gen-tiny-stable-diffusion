import argparse
import csv
import gc
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import DownloadConfig, Features, Image as HFImage, Value, load_dataset, Dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from tqdm import tqdm

# ------------------------------------------------------------------ #
# 1) Dataset loader (creates a lean two‑column dataset: “caption”, “image_id”) #
# ------------------------------------------------------------------ #
def load_hf_dataset(
    n: int = 1_000,
    cfg: str = "2014",
    seed: int = 0,
    streaming: bool = False,
):
    print(
        f"⇢  Loading COCO/{cfg} "
        f"{'streaming' if streaming else 'download'} mode…"
    )

    if cfg == "2014":
        features = Features(
            {
                "image": HFImage(),
                "image_id": Value("int64"),
                "caption_id": Value("int64"),
                "sentences": {
                    "raw": Value("string"),
                    "tokens": [Value("string")],
                    "token_ids": [Value("int64")],
                },
            }
        )
    else:  # 2017, etc.
        features = Features(
            {
                "image": HFImage(),
                "image_id": Value("int64"),
                "caption_id": Value("int64"),
                "sentences_raw": [Value("string")],
            }
        )

    ds = load_dataset(
        "HuggingFaceM4/COCO",
        name=cfg,
        split="train",
        trust_remote_code=True,
        streaming=streaming,
        download_config=DownloadConfig(resume_download=True, max_retries=10),
        features=features,
    )

    # --------- pull caption & image_id, drop everything else --------
    def _add_caption_and_id(example):
        if cfg == "2014":
            example["caption"] = example["sentences"]["raw"]
        else:
            example["caption"] = example["sentences_raw"][0]
        return {"caption": example["caption"], "image_id": example["image_id"]}

    final_features = Features({
        "caption": Value("string"),
        "image_id": Value("int64"),
    })

    ds = ds.map(
        _add_caption_and_id,
        remove_columns=[col for col in ds.column_names if col not in ['caption', 'image_id']],
        features=final_features,
        desc="Building `caption` & `image_id` columns",
    )

    if streaming:
        ds_list = list(ds.take(n))
        ds = Dataset.from_pandas(pd.DataFrame(ds_list), features=final_features)
        print(f"✅ Streamed and collected {len(ds)} samples.")
    else:
        ds = ds.shuffle(seed=seed)
        if n < len(ds):
            ds = ds.select(range(n))

    print(f"✅ Dataset ready → columns: {ds.column_names} (len = {len(ds)})")
    return ds


# ------------------------------------------------------------------ #
# 2) Embedding helpers                                               #
# ------------------------------------------------------------------ #
def generate_embeddings(
    captions,
    model_name="all-MiniLM-L6-v2",
    batch_size=128,
):
    if not captions:
        return np.empty((0,))

    model = SentenceTransformer(
        model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print(f"⇢ Encoding {len(captions):,} captions on {model.device}…")

    embeddings = model.encode(
        captions,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"✅ Embeddings shape: {embeddings.shape}")
    return embeddings


def yield_embeddings_stream(
    dataset,
    model_name="all-MiniLM-L6-v2",
    batch_size=128,
):
    """
    Generator that yields embedding numpy arrays (batch_size × dim)
    without materialising all captions at once.
    """
    model = SentenceTransformer(
        model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    buf = []
    for ex in dataset:
        buf.append(ex["caption"])
        if len(buf) == batch_size:
            yield model.encode(buf, convert_to_numpy=True)
            buf.clear()
    if buf:  # leftovers
        yield model.encode(buf, convert_to_numpy=True)


# ------------------------------------------------------------------ #
# 3) K‑Means                                                         #
# ------------------------------------------------------------------ #
def perform_kmeans(embeddings, k, seed=42):
    if embeddings.size == 0:
        raise ValueError("No embeddings to cluster.")

    if embeddings.shape[0] < k:
        k = embeddings.shape[0]
        print(f"⚠️  Reduced k to {k} (fewer samples than clusters).")

    print(f"⇢ Running K‑Means (k = {k})…")
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    km.fit(embeddings)
    print("✅ K‑Means done.")
    return km.labels_


# ------------------------------------------------------------------ #
# 4) CLI / main                                                      #
# ------------------------------------------------------------------ #
def main():
    p = argparse.ArgumentParser(
        description="Cluster COCO captions with Sentence‑Transformer + K‑Means."
    )
    p.add_argument("--num_samples", type=int, default=50_000)
    p.add_argument("--dataset_cfg", default="2014")
    p.add_argument("--streaming", action="store_true")
    p.add_argument("--streaming_embeddings", action="store_true")
    p.add_argument("--num_clusters", type=int, default=50)
    p.add_argument("--embedding_model", default="all-MiniLM-L6-v2")
    p.add_argument("--embedding_batch_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output_csv",
        default="coco_caption_clusters.csv",
        help="File to save <image_id,caption,cluster_id> pairs.",
    )
    args = p.parse_args()

    print("-" * 60)
    print("Run parameters:")
    for k, v in vars(args).items():
        print(f"  {k:20} = {v}")
    print("-" * 60)

    # 1) dataset
    ds = load_hf_dataset(
        n=args.num_samples,
        cfg=args.dataset_cfg,
        seed=args.seed,
        streaming=args.streaming,
    )

    # 2) embeddings
    # Store image_ids along with captions, especially for streaming modes
    all_data = [{"caption": ex["caption"], "image_id": ex["image_id"]} for ex in ds]
    captions = [item["caption"] for item in all_data]
    image_ids = [item["image_id"] for item in all_data]

    if args.streaming_embeddings:
        print("⇢ Generating embeddings (streaming mode)...")
        emb_chunks = list(
            yield_embeddings_stream(
                ({"caption": cap} for cap in captions),
                model_name=args.embedding_model,
                batch_size=args.embedding_batch_size,
            )
        )
        embeddings = np.vstack(emb_chunks)
        print(f"✅ Embeddings shape: {embeddings.shape}")
    else:
        embeddings = generate_embeddings(
            captions,
            model_name=args.embedding_model,
            batch_size=args.embedding_batch_size,
        )

    # 3) K‑Means
    labels = perform_kmeans(
        embeddings,
        k=args.num_clusters,
        seed=args.seed,
    )

    # 4) save CSV
    out_path = Path(args.output_csv)
    print(f"⇢ Saving results → {out_path.resolve()}")
    df_results = pd.DataFrame({
        "image_id": image_ids,
        "caption": captions,
        "cluster_id": labels
    })
    df_results.to_csv(
        out_path, index=False, quoting=csv.QUOTE_ALL
    )
    print("✅ CSV saved.")

    # 5) quick diagnostics
    vc = (
        pd.Series(labels, name="cluster")
        .value_counts()
        .sort_index()
    )
    print("\nCluster counts:")
    print(vc.to_string())
    print("-" * 60)
    print("✓ All done.")


if __name__ == "__main__":
    warnings.simplefilter("default")
    torch.set_grad_enabled(False)
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
