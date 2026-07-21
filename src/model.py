import torch
import torch.nn as nn

class LeafDiseaseCNN(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

model = LeafDiseaseCNN(num_classes=4)
x = torch.randn(8, 3, 224, 224)
logits = model(x)
print(logits.shape)  # torch.Size([8, 4])
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
assert logits.shape == (8, 4), "Output shape is incorrect!"
print("Output shape verified successfully!")