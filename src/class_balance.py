import csv
import os
from collections import Counter

from src.dataloader import train_dataset, CLASS_NAMES

# Get labels from the complete dataset
labels = [label for _, label in train_dataset.samples]
counts = Counter(labels)

# Create reports folder if needed
os.makedirs("reports", exist_ok=True)

output_path = "reports/class_balance.csv"

# Save class distribution
with open(output_path, "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow(["class", "count"])

    for idx, class_name in enumerate(CLASS_NAMES):
        writer.writerow([
            class_name,
            counts.get(idx, 0)
        ])

print("\nClass Distribution:")

for idx, class_name in enumerate(CLASS_NAMES):
    print(f"{class_name}: {counts.get(idx, 0)} images")

print(f"\nSaved class distribution to {output_path}")