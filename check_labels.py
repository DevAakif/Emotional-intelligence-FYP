
import pickle
import numpy as np

with open("aligned_50.pkl", "rb") as f:
    data = pickle.load(f, encoding="latin1")

for split_name in ["train", "valid", "test"]:
    split = data[split_name]

    labels = split["classification_labels"]
    annotations = split["annotations"]
    ids = split["id"]
    texts = split["raw_text"]

    print(f"\n--- {split_name.upper()} ---")

    for i, (label, annotation) in enumerate(
        zip(labels, annotations)
    ):
        expected = {
            0: "Negative",
            1: "Neutral",
            2: "Positive",
        }.get(int(label))

        if annotation != expected:
            print("Index:", i)
            print("ID:", ids[i])
            print("Text:", texts[i])
            print("Classification label:", label)
            print("Annotation:", annotation)
