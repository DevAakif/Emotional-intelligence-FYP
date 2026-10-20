
import pickle
from collections import defaultdict

with open("aligned_50.pkl", "rb") as f:
    data = pickle.load(f, encoding="latin1")

for split_name in ["train", "valid", "test"]:
    split = data[split_name]
    counts = defaultdict(int)

    for label, annotation in zip(
        split["classification_labels"],
        split["annotations"]
    ):
        counts[(float(label), str(annotation))] += 1

    print(f"\n{split_name.upper()} — label mapping")
    for (label, annotation), count in sorted(counts.items()):
        print(
            f"Classification label: {label:.0f} | "
            f"Annotation: {annotation} | Count: {count}"
        )
