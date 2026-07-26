from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

from src.transforms import train_transform, IMAGENET_MEAN, IMAGENET_STD


# Find one leaf image from the training dataset
image_files = list(Path("data/train").glob("*/*.jpg"))

if not image_files:
    raise FileNotFoundError("No JPG images found inside data/train.")

image_path = image_files[0]
image = Image.open(image_path).convert("RGB")

mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

fig, axes = plt.subplots(2, 4, figsize=(12, 6))

for ax in axes.flat:
    augmented = train_transform(image)

    # Undo normalization for visualization
    augmented = augmented * std + mean
    augmented = torch.clamp(augmented, 0, 1)

    ax.imshow(augmented.permute(1, 2, 0))
    ax.axis("off")

plt.suptitle("8 Augmented Versions of One Leaf")
plt.tight_layout()

Path("reports").mkdir(exist_ok=True)
plt.savefig("reports/augment_samples.png", dpi=150, bbox_inches="tight")

print("Saved: reports/augment_samples.png")

plt.show()