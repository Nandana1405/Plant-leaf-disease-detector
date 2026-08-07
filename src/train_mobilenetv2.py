import copy
import os
import matplotlib.pyplot as plt
import torch

from torch import nn, optim
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import recall_score

from torchvision import models
import torch.nn as nn
from src.dataloader import train_loader, val_loader


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# MODEL
# --------------------------------------------------

weights = models.MobileNet_V2_Weights.IMAGENET1K_V1

model = models.mobilenet_v2(weights=weights)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
model.classifier[1] = nn.Linear(model.last_channel, 4)

# Train only classifier
for param in model.classifier.parameters():
    param.requires_grad = True

model = model.to(device)

# --------------------------------------------------
# CLASS WEIGHTS
# --------------------------------------------------

# Get labels only from the training split
train_labels = [
    train_loader.dataset.dataset.samples[i][1]
    for i in train_loader.dataset.indices
]

class_counts = Counter(train_labels)

num_classes = 4
total_samples = len(train_labels)

class_weights = [
    total_samples / (num_classes * class_counts[i])
    for i in range(num_classes)
]

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(device)

print("Training class counts:", class_counts)
print("Class weights:", class_weights)


# --------------------------------------------------
# LOSS + OPTIMIZER
# --------------------------------------------------

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = optim.Adam(
    model.classifier.parameters(),
    lr=1e-3
)


# --------------------------------------------------
# TRAIN ONE EPOCH
# --------------------------------------------------

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):
    model.train()

    total_loss = 0.0

    for images, labels in tqdm(
        loader,
        desc="Train"
    ):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item() * images.size(0)
        )

    return total_loss / len(loader.dataset)


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device
):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        total_loss += (
            loss.item() * images.size(0)
        )

        preds = outputs.argmax(dim=1)

        correct += (
            preds == labels
        ).sum().item()

        total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


# --------------------------------------------------
# PER-CLASS RECALL
# --------------------------------------------------

@torch.no_grad()
def per_class_recall(
    model,
    loader,
    device
):
    model.eval()

    all_labels = []
    all_preds = []

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        preds = outputs.argmax(dim=1)

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_preds.extend(
            preds.cpu().numpy()
        )

    recalls = recall_score(
        all_labels,
        all_preds,
        labels=[0, 1, 2, 3],
        average=None,
        zero_division=0
    )

    return recalls


# --------------------------------------------------
# EARLY STOPPING SETTINGS
# --------------------------------------------------

best_val = float("inf")

patience = 2

wait = 0

best_weights = None

best_epoch = 0
best_val_acc = 0.0


train_losses = []
val_losses = []


# --------------------------------------------------
# TRAINING LOOP
# --------------------------------------------------

# Maximum 5 epochs.
# Early stopping may stop training sooner.

for epoch in range(1, 6):

    train_loss = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    val_loss, val_acc = validate(
        model,
        val_loader,
        criterion,
        device
    )

    train_losses.append(
        train_loss
    )

    val_losses.append(
        val_loss
    )

    print(
        f"Epoch {epoch} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )


    # Check whether validation loss improved

    if val_loss < best_val:

        best_val = val_loss

        best_val_acc = val_acc

        best_epoch = epoch

        wait = 0

        best_weights = copy.deepcopy(
            model.state_dict()
        )

    else:

        wait += 1

        print(
            f"No improvement. "
            f"Patience: {wait}/{patience}"
        )

        if wait >= patience:

            print(
                "Early stopping triggered!"
            )

            break


# --------------------------------------------------
# RESTORE BEST MODEL
# --------------------------------------------------

if best_weights is not None:

    model.load_state_dict(
        best_weights
    )

    print(
        f"\nBest model restored "
        f"from epoch {best_epoch}"
    )


# --------------------------------------------------
# PER-CLASS RECALL
# --------------------------------------------------

recalls = per_class_recall(
    model,
    val_loader,
    device
)

class_names = [
    "healthy",
    "early_blight",
    "late_blight",
    "leaf_mold"
]

print("\nPer-Class Recall:")

for class_name, recall in zip(
    class_names,
    recalls
):
    print(
        f"{class_name}: "
        f"{recall:.4f}"
    )


# --------------------------------------------------
# BEST METRICS
# --------------------------------------------------

print(
    f"\nBest Epoch: {best_epoch}"
)

print(
    f"Best Validation Loss: "
    f"{best_val:.4f}"
)

print(
    f"Best Validation Accuracy: "
    f"{best_val_acc:.4f}"
)


# --------------------------------------------------
# CREATE FOLDERS
# --------------------------------------------------

os.makedirs(
    "models",
    exist_ok=True
)

os.makedirs(
    "reports",
    exist_ok=True
)


# --------------------------------------------------
# SAVE BEST MODEL
# --------------------------------------------------

torch.save(
    model.state_dict(),
    "models/leaf_cnn_best.pth"
)


# --------------------------------------------------
# SAVE TRAINING CURVE
# --------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

epochs = range(
    1,
    len(train_losses) + 1
)

plt.plot(
    epochs,
    train_losses,
    label="Train Loss"
)

plt.plot(
    epochs,
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "Training vs Validation Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "reports/training_curves.png"
)

plt.show()


# --------------------------------------------------
# FINISHED
# --------------------------------------------------

print(
    "\nBest model saved to "
    "models/leaf_cnn_best.pth"
)

print(
    "Loss curve saved to "
    "reports/training_curves.png"
)
import os

os.makedirs("models", exist_ok=True)

torch.save(model.state_dict(),
           "models/mobilenetv2_leaf_best.pth")

print("Model saved to models/mobilenetv2_leaf_best.pth")