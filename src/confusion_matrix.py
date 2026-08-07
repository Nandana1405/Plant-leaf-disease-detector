import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from torchvision import models

from src.dataloader import val_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class names
with open("models/class_names.json", "r") as f:
    class_names = json.load(f)

NUM_CLASSES = len(class_names)

# Load ResNet18
weights = models.ResNet18_Weights.IMAGENET1K_V1
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

checkpoint = torch.load(
    "models/resnet18_best.pth",
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in val_loader:

        images = images.to(device)

        outputs = model(images)

        preds = outputs.argmax(dim=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)

os.makedirs("reports", exist_ok=True)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(cmap="Greens", xticks_rotation=45)

plt.tight_layout()

plt.savefig("reports/confusion_matrix.png")

plt.show()

print(classification_report(
    y_true,
    y_pred,
    target_names=class_names
))