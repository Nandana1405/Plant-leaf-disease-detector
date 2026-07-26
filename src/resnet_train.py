import os
import json
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

from src.dataloader import train_loader, val_loader


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

CLASS_NAMES = [
    "healthy",
    "early_blight",
    "late_blight",
    "leaf_mold"
]

NUM_CLASSES = len(CLASS_NAMES)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# LOAD PRETRAINED RESNET18
# --------------------------------------------------

weights = models.ResNet18_Weights.IMAGENET1K_V1
model = models.resnet18(weights=weights)

# Freeze entire pretrained backbone
for param in model.parameters():
    param.requires_grad = False

# Replace final classifier
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, NUM_CLASSES)

model = model.to(device)

criterion = nn.CrossEntropyLoss()


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

@torch.no_grad()
def validate(model, loader):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    val_loss = total_loss / total
    val_acc = correct / total

    return val_loss, val_acc


# --------------------------------------------------
# TRAIN ONE EPOCH
# --------------------------------------------------

def train_one_epoch(model, loader, optimizer):

    model.train()

    total_loss = 0.0

    for images, labels in tqdm(loader, desc="Training"):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


# --------------------------------------------------
# SAVE BEST MODEL
# --------------------------------------------------

os.makedirs("models", exist_ok=True)

best_val_acc = 0.0
best_weights = copy.deepcopy(model.state_dict())
best_epoch = 0

start_time = time.time()


# ==================================================
# PHASE 1
# Train only FC head
# ==================================================

print("\n====================================")
print("PHASE 1: Training classifier head")
print("====================================")

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=1e-3
)

# Keep this small because CPU training can be slow
PHASE1_EPOCHS = 2

for epoch in range(1, PHASE1_EPOCHS + 1):

    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer
    )

    val_loss, val_acc = validate(
        model,
        val_loader
    )

    print(
        f"\nPhase 1 Epoch {epoch} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    if val_acc > best_val_acc:

        best_val_acc = val_acc
        best_weights = copy.deepcopy(model.state_dict())
        best_epoch = epoch

        print("New best model!")


# ==================================================
# PHASE 2
# Unfreeze layer4 + classifier
# ==================================================

print("\n====================================")
print("PHASE 2: Fine-tuning layer4")
print("====================================")

for param in model.layer4.parameters():
    param.requires_grad = True


# Different learning rates:
# smaller LR for pretrained layer4
# larger LR for new classifier
optimizer = optim.Adam(
    [
        {
            "params": model.layer4.parameters(),
            "lr": 1e-4
        },
        {
            "params": model.fc.parameters(),
            "lr": 5e-4
        }
    ]
)

PHASE2_EPOCHS = 2

for epoch in range(1, PHASE2_EPOCHS + 1):

    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer
    )

    val_loss, val_acc = validate(
        model,
        val_loader
    )

    print(
        f"\nPhase 2 Epoch {epoch} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    if val_acc > best_val_acc:

        best_val_acc = val_acc
        best_weights = copy.deepcopy(model.state_dict())

        best_epoch = PHASE1_EPOCHS + epoch

        print("New best model!")


# --------------------------------------------------
# RESTORE BEST MODEL
# --------------------------------------------------

model.load_state_dict(best_weights)


# --------------------------------------------------
# SAVE CHECKPOINT
# --------------------------------------------------

checkpoint = {
    "model_state_dict": model.state_dict(),
    "epoch": best_epoch,
    "val_acc": best_val_acc,
    "class_names": CLASS_NAMES
}

torch.save(
    checkpoint,
    "models/resnet18_leaf_best.pth"
)


# --------------------------------------------------
# SAVE CLASS NAMES JSON
# --------------------------------------------------

with open(
    "models/class_names.json",
    "w"
) as f:

    json.dump(
        CLASS_NAMES,
        f,
        indent=4
    )


# --------------------------------------------------
# TRAINING TIME
# --------------------------------------------------

end_time = time.time()

training_time = end_time - start_time

minutes = int(training_time // 60)
seconds = int(training_time % 60)


# --------------------------------------------------
# FINAL RESULTS
# --------------------------------------------------

print("\n====================================")
print("TRAINING COMPLETE")
print("====================================")

print(f"Best Epoch: {best_epoch}")
print(
    f"Best Validation Accuracy: "
    f"{best_val_acc:.4f} "
    f"({best_val_acc * 100:.2f}%)"
)

print(
    f"Training Time: "
    f"{minutes} min {seconds} sec"
)

print("Hardware:", device)

print(
    "\nCheckpoint saved to "
    "models/resnet18_leaf_best.pth"
)

print(
    "Class names saved to "
    "models/class_names.json"
)