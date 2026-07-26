import torch
import torch.nn as nn
from torchvision import models

from src.dataloader import train_loader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# -------------------------------------------------
# 1. LOAD PRETRAINED RESNET18
# -------------------------------------------------

weights = models.ResNet18_Weights.IMAGENET1K_V1
model = models.resnet18(weights=weights)


# -------------------------------------------------
# 2. FREEZE PRETRAINED BACKBONE
# -------------------------------------------------

for param in model.parameters():
    param.requires_grad = False


# -------------------------------------------------
# 3. REPLACE CLASSIFICATION HEAD
# -------------------------------------------------

num_classes = 4

in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

model = model.to(device)


# -------------------------------------------------
# 4. COUNT PARAMETERS
# -------------------------------------------------

trainable = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

total = sum(
    p.numel()
    for p in model.parameters()
)

print("\nParameter Count")
print("Trainable parameters:", trainable)
print("Total parameters:", total)


# -------------------------------------------------
# 5. PRINT AND SAVE LAYER NAMES
# -------------------------------------------------

print("\nResNet18 Layers:")

layer_names = []

for name, module in model.named_children():
    print(name)
    layer_names.append(name)

with open("day9_layer_names.txt", "w") as f:
    for name in layer_names:
        f.write(name + "\n")

print("\nLayer names saved to day9_layer_names.txt")


# -------------------------------------------------
# 6. FORWARD PASS USING LEAF BATCH
# -------------------------------------------------

images, labels = next(iter(train_loader))

images = images.to(device)

model.eval()

with torch.no_grad():
    outputs = model(images)

print("\nForward Pass Successful!")
print("Input batch shape:", images.shape)
print("Output shape:", outputs.shape)