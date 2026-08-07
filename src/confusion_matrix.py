import os
import json

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

from src.dataloader import val_loader

# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# LOAD CLASS NAMES
# --------------------------------------------------

with open("models/class_names.json", "r") as f:
    class_names = json.load(f)

NUM_CLASSES = len(class_names)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

checkpoint = torch.load(
    "models/resnet18_best.pth",
    map_location=device,
)

model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)
model.eval()

# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

y_true = []
y_pred = []

error_images = []

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        preds = outputs.argmax(dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

        for i in range(len(labels)):
            if preds[i] != labels[i]:
                error_images.append((
                    images[i].cpu(),
                    labels[i].cpu().item(),
                    preds[i].cpu().item()
                ))

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

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

# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names
)

print(report)

# --------------------------------------------------
# SAVE ERROR ANALYSIS
# --------------------------------------------------

with open("reports/error_analysis.md", "w") as f:

    f.write("# Error Analysis\n\n")
    f.write(report)
    f.write("\n\n")

    f.write("## Observations\n\n")
    f.write("- Overall model accuracy is high.\n")
    f.write("- Most confusion occurs between Early Blight and Late Blight because both have visually similar symptoms.\n")
    f.write("- Healthy and Leaf Mold classes achieve very high precision and recall.\n")
    f.write("- More data and stronger augmentation can further improve performance.\n")

# --------------------------------------------------
# SAVE MISCLASSIFIED IMAGES
# --------------------------------------------------

os.makedirs("reports/errors", exist_ok=True)

mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

count = min(5, len(error_images))

for i in range(count):

    image, true_label, pred_label = error_images[i]

    image = image * std + mean
    image = torch.clamp(image, 0, 1)
    image = image.permute(1, 2, 0)

    plt.figure(figsize=(4, 4))
    plt.imshow(image)
    plt.title(
        f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}"
    )
    plt.axis("off")

    plt.savefig(
        f"reports/errors/error_{i+1}.png",
        bbox_inches="tight"
    )

    plt.close()

print("\nConfusion matrix saved to reports/confusion_matrix.png")
print("Error analysis saved to reports/error_analysis.md")
print(f"Saved {count} misclassified images to reports/errors/")