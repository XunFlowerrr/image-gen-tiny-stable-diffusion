import pandas as pd

def preview_clusters(
    csv_path: str,
    num_samples: int = 3,
    random_state: int = 42
):
    """
    Load <imgid,caption,cluster_id> pairs from csv_path and
    print `num_samples` random imgids and captions for each cluster.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return
    except Exception as e:
        print(f"Error loading CSV {csv_path}: {e}")
        return

    id_column = 'imgid'
    if id_column not in df.columns:
        print(f"Warning: '{id_column}' column not found in {csv_path}. Checking for 'image_id'...")
        if 'image_id' in df.columns:
             print("Found 'image_id' instead. Using it as fallback.")
             id_column = 'image_id'
        else:
             print(f"Warning: Neither '{id_column}' nor 'image_id' found. Cannot display image IDs.")
             df[id_column] = 'N/A'


    clusters = sorted(df['cluster_id'].unique())

    for cid in clusters:
        sub = df[df['cluster_id'] == cid]
        count = len(sub)
        if count == 0:
            continue

        # Sample the subset DataFrame
        samples_df = sub.sample(
            n=min(num_samples, count),
            random_state=random_state
        )

        print(f"\n--- Cluster {cid} ({count} captions) ---")
        for i, row in enumerate(samples_df.itertuples(), 1):
            img_id = getattr(row, id_column, 'N/A')
            caption = getattr(row, 'caption', 'N/A')
            print(f" {i}. [Image ID: {img_id}] {caption}")

if __name__ == "__main__":
    preview_clusters("coco_caption_clusters.csv", num_samples=5)
