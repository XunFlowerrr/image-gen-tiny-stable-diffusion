import pandas as pd
import matplotlib.pyplot as plt

# 1) Define your cluster → short-label mapping
CLUSTER_LABELS = {
    0:  "fire hydrants",
    1:  "individual portraits",
    2:  "giraffes",
    3:  "video gaming",
    4:  "women & cooking",
    5:  "people interactions",
    6:  "large trucks",
    7:  "beach scenes",
    8:  "fruit & market stalls",
    9:  "horses & riders",
    10: "dogs in vehicles",
    11: "trains",
    12: "skiing & snow sports",
    13: "prepared food dishes",
    14: "luggage & travel",
    15: "motorcycles",
    16: "women’s tennis",
    17: "urban street views",
    18: "airplanes",
    19: "zebras",
    20: "children & toiletries",
    21: "surfing",
    22: "living rooms",
    23: "vases & decor",
    24: "boats & ships",
    25: "kite flying",
    26: "cows",
    27: "cakes & celebrations",
    28: "public restrooms",
    29: "buses",
    30: "road signs",
    31: "Frisbee play",
    32: "clocks & towers",
    33: "bears & stuffed toys",
    34: "mobile phones",
    35: "bathrooms",
    36: "cats",
    37: "beds",
    38: "skateboarding",
    39: "birds & ponds",
    40: "sheep & goats",
    41: "men’s tennis",
    42: "baseball",
    43: "motorcycles & bicycles",
    44: "kitchens",
    45: "elephants",
    46: "umbrellas & rain",
    47: "pizza",
    48: "benches",
    49: "computers & office"
}

def preview_and_plot_clusters(
    csv_path: str,
    num_samples: int = 3,
    random_state: int = 42,
    plot_path: str = "cluster_counts.png"
):
    """
    1) Load <imgid,caption,cluster_id> pairs
    2) Map cluster_id → human-readable label
    3) Print a few example imgids and captions per cluster
    4) Plot and save a bar chart of counts by cluster label
    """
    # Load - check for imgid
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

    df['cluster_label'] = df['cluster_id'].map(CLUSTER_LABELS)

    # Preview samples
    for cid, label in CLUSTER_LABELS.items():
        sub = df[df['cluster_id'] == cid]
        count = len(sub)
        if count == 0:
            continue

        # Sample the subset DataFrame to keep ID and caption together
        samples_df = sub.sample(
            n=min(num_samples, count),
            random_state=random_state
        )

        print(f"\n--- Cluster {cid}: {label} ({count} captions) ---")
        for i, row in enumerate(samples_df.itertuples(), 1):
            # Access imgid (or fallback) and caption from the sampled row
            img_id = getattr(row, id_column, 'N/A')
            caption = getattr(row, 'caption', 'N/A')
            print(f" {i}. [Image ID: {img_id}] {caption}")

    # Plot counts
    counts = df['cluster_label'].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    counts.plot.bar()
    plt.title("Number of Captions per Cluster")
    plt.ylabel("Caption Count")
    plt.xlabel("Cluster Label")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save to file
    plt.savefig(plot_path)
    print(f"\n✅ Plot saved to '{plot_path}'")

    plt.show()


if __name__ == "__main__":
    preview_and_plot_clusters(
        "coco_caption_clusters.csv",
        num_samples=5,
        plot_path="cluster_counts.png"
    )
