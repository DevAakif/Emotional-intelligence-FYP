
import os
import pickle
import numpy as np
import pandas as pd

# Load the processed CMU-MOSI dataset
with open("aligned_50.pkl", "rb") as f:
    data = pickle.load(f, encoding="latin1")

# Create an output directory
output_dir = "mosi_export"
os.makedirs(output_dir, exist_ok=True)

for split_name in ["train", "valid", "test"]:
    split = data[split_name]

    # Export readable metadata
    metadata = pd.DataFrame({
        "id": split["id"],
        "raw_text": split["raw_text"],
        "annotation": split["annotations"],
        "classification_label": split["classification_labels"],
        "regression_label": split["regression_labels"],
    })

    csv_path = os.path.join(
        output_dir, f"{split_name}_metadata.csv"
    )
    metadata.to_csv(csv_path, index=False)

    # Export numerical features without losing their dimensions
    npz_path = os.path.join(
        output_dir, f"{split_name}_features.npz"
    )

    np.savez_compressed(
        npz_path,
        text=split["text"],
        audio=split["audio"],
        vision=split["vision"],
        text_bert=split["text_bert"],
        ids=np.asarray(split["id"]),
        classification_labels=split["classification_labels"],
        regression_labels=split["regression_labels"],
    )

    print(f"\nExported {split_name}:")
    print(f"  Metadata: {csv_path}")
    print(f"  Features: {npz_path}")
    print("  Samples:", len(metadata))
    print("  Text features:", split["text"].shape)
    print("  Audio features:", split["audio"].shape)
    print("  Visual features:", split["vision"].shape)

print("\nExport complete.")
