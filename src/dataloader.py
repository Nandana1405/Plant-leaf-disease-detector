import matplotlib.pyplot as plt
import torchvision
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from collections import Counter

CLASS_NAMES = ["healthy", "early_blight", "late_blight", "leaf_mold"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


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
                self.samples.append((img_path, CLASS_TO_IDX[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


dataset = LeafDiseaseDataset("data/train", transform=transform)

labels = [label for _, label in dataset.samples]
counts = Counter(labels)

print("Class Distribution:")
for class_name, idx in CLASS_TO_IDX.items():
    print(f"{class_name}: {counts[idx]} images")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_ds, val_ds = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_ds,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

images, labels = next(iter(train_loader))
print(images.shape)

grid = torchvision.utils.make_grid(images[:8], nrow=4)

mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

grid = grid * std + mean
grid = torch.clamp(grid, 0, 1)
grid = grid.permute(1, 2, 0)

plt.figure(figsize=(10, 6))
plt.imshow(grid)
plt.axis("off")
plt.title("Sample Training Images")
plt.show()