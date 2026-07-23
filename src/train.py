import copy
import os
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from tqdm import tqdm

from src.model import LeafDiseaseCNN
from src.dataloader import train_loader, val_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LeafDiseaseCNN(num_classes=4).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0

    for images, labels in tqdm(loader, desc="Train"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
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

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


best_val = float("inf")
patience = 3
wait = 0
best_weights = None

train_losses = []
val_losses = []

for epoch in range(1, 11):

    train_loss = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
    )

    val_loss, val_acc = validate(
        model,
        val_loader,
        criterion,
        device,
    )

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(
        f"Epoch {epoch} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    if val_loss < best_val:
        best_val = val_loss
        wait = 0
        best_weights = copy.deepcopy(model.state_dict())
    else:
        wait += 1

        if wait >= patience:
            print("Early stopping triggered!")
            break


if best_weights is not None:
    model.load_state_dict(best_weights)

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

torch.save(model.state_dict(), "models/leaf_cnn_best.pth")

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)

plt.savefig("reports/training_curves.png")
plt.show()

print("Best model saved to models/leaf_cnn_best.pth")
print("Loss curve saved to reports/training_curves.png")