import os
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import models

from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
)

from src.dataloader import val_loader

# -----------------------------------------
# DEVICE
# -----------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------
# LOAD CLASS NAMES
# -----------------------------------------

with open("models/class_names.json", "r") as f:
    class_names = json.load(f)

NUM_CLASSES = len(class_names)

# -----------------------------------------
# LOAD MODEL
# -----------------------------------------

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

checkpoint = torch.load(
    "models/resnet18_best.pth",
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# -----------------------------------------
# INFERENCE
# -----------------------------------------

y_true = []
y_pred = []
all_probs = []

softmax = nn.Softmax(dim=1)

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)

        outputs = model(images)

        probs = softmax(outputs)

        preds = probs.argmax(dim=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_probs = np.array(all_probs)

# -----------------------------------------
# CLASSIFICATION REPORT
# -----------------------------------------

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names
)

print(report)

os.makedirs("reports", exist_ok=True)

with open("reports/classification_report.txt", "w") as f:
    f.write(report)

# -----------------------------------------
# BINARY METRICS
# healthy = 0
# disease = 1
# -----------------------------------------

healthy_index = class_names.index("healthy")

y_binary = np.array([
    0 if x == healthy_index else 1
    for x in y_true
])

disease_probs = 1 - all_probs[:, healthy_index]

precision, recall, thresholds = precision_recall_curve(
    y_binary,
    disease_probs
)

chosen_threshold = 0.5

for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
    if r >= 0.95:
        chosen_threshold = float(t)
        break

print("Chosen Threshold:", chosen_threshold)

# -----------------------------------------
# PR CURVE
# -----------------------------------------

plt.figure(figsize=(6,5))
plt.plot(recall, precision)

idx = np.argmin(np.abs(thresholds - chosen_threshold))

plt.scatter(
    recall[idx],
    precision[idx],
    s=80,
    label=f"Threshold={chosen_threshold:.2f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()

plt.tight_layout()

plt.savefig("reports/pr_curve.png")

# -----------------------------------------
# SAVE THRESHOLD
# -----------------------------------------

config = {
    "healthy_vs_disease_threshold": chosen_threshold
}

with open("models/inference_config.json", "w") as f:
    json.dump(config, f, indent=4)

# -----------------------------------------
# SLA
# -----------------------------------------

sla = """
# Polyhouse Deployment Recommendation

Recommended threshold:
Maintain recall >= 0.95 for diseased plants.

Reason:
Missing diseased plants is more harmful than producing a few false alarms.

Recommended operating threshold:
{}

Deploy this threshold for production inference.
""".format(chosen_threshold)

with open("reports/sla_recommendation.md", "w") as f:
    f.write(sla)

print("\nSaved:")
print("reports/classification_report.txt")
print("reports/pr_curve.png")
print("reports/sla_recommendation.md")
print("models/inference_config.json")