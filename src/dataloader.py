import matplotlib.pyplot as plt
import torchvision
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from collections import Counter

from src.transforms import train_transform, val_transform


CLASS_NAMES = ["healthy", "early_blight", "late_blight", "leaf_mold"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


class LeafDiseaseDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        for class_name in CLASS_NAMES:
            class_dir = self.root / class_name

            if not class_dir.exists():
                continue

            for img_path in class_dir.glob("*.jpg"):
                self.samples.append(
                    (img_path, CLASS_TO_IDX[class_name])
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# Training dataset uses augmentation
train_dataset = LeafDiseaseDataset(
    "data/train",
    transform=train_transform
)

# Validation dataset does NOT use random augmentation
val_dataset = LeafDiseaseDataset(
    "data/train",
    transform=val_transform
)


# Count class distribution
labels = [label for _, label in train_dataset.samples]
counts = Counter(labels)

print("Class Distribution:")
for class_name, idx in CLASS_TO_IDX.items():
    print(f"{class_name}: {counts[idx]} images")


# Create the same reproducible 80/20 split
dataset_size = len(train_dataset)

generator = torch.Generator().manual_seed(42)
indices = torch.randperm(
    dataset_size,
    generator=generator
).tolist()

train_size = int(0.8 * dataset_size)

train_indices = indices[:train_size]
val_indices = indices[train_size:]


# Same split, but different transforms
train_ds = Subset(
    train_dataset,
    train_indices
)

val_ds = Subset(
    val_dataset,
    val_indices
)


# DataLoaders
train_loader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

val_loader = DataLoader(
    val_ds,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)


# Verify one training batch
images, labels = next(iter(train_loader))

print("Training batch shape:", images.shape)
print("Training samples:", len(train_ds))
print("Validation samples:", len(val_ds))


# Visualize augmented training images
grid = torchvision.utils.make_grid(images[:8], nrow=4)

mean = torch.tensor(
    [0.485, 0.456, 0.406]
).view(3, 1, 1)

std = torch.tensor(
    [0.229, 0.224, 0.225]
).view(3, 1, 1)

# Undo normalization for display
grid = grid * std + mean
grid = torch.clamp(grid, 0, 1)
grid = grid.permute(1, 2, 0)

plt.figure(figsize=(10, 6))
plt.imshow(grid)
plt.axis("off")
plt.title("Augmented Training Images")
plt.show()