import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 4

weights = models.MobileNet_V2_Weights.IMAGENET1K_V1

model = models.mobilenet_v2(weights=weights)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
model.classifier[1] = nn.Linear(
    model.last_channel,
    NUM_CLASSES
)

# Unfreeze classifier
for param in model.classifier.parameters():
    param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())

print(f"Trainable Parameters: {trainable:,}")
print(f"Total Parameters: {total:,}")