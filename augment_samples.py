import os
import matplotlib.pyplot as plt
from PIL import Image

from src.transforms import train_transform
from src.dataloader import train_dataset


# Get one original image
image_path, label = train_dataset.samples[0]
original = Image.open(image_path).convert("RGB")

os.makedirs("reports", exist_ok=True)

fig, axes = plt.subplots(2, 4, figsize=(12, 6))

for ax in axes.flat:
    augmented = train_transform(original)

    # Undo normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for channel, m, s in zip(augmented, mean, std):
        channel.mul_(s).add_(m)

    augmented = augmented.clamp(0, 1)
    augmented = augmented.permute(1, 2, 0)

    ax.imshow(augmented)
    ax.axis("off")

plt.suptitle("8 Augmented Versions of One Leaf")
plt.tight_layout()

plt.savefig(
    "reports/augment_samples.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("Saved to reports/augment_samples.png")