import os
import time
import torch
import torch.nn as nn
from torchvision import models

device = torch.device("cpu")

# Load MobileNetV2
model = models.mobilenet_v2(
    weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
)

# Replace classifier
model.classifier[1] = nn.Linear(
    model.last_channel,
    4
)

# Load trained weights
model.load_state_dict(
    torch.load(
        "models/mobilenetv2_leaf_best.pth",
        map_location=device
    )
)

model.eval()

# Dummy input
dummy = torch.randn(1, 3, 224, 224)

# Warm-up
with torch.no_grad():
    for _ in range(10):
        model(dummy)

# Benchmark
start = time.perf_counter()

with torch.no_grad():
    for _ in range(100):
        model(dummy)

end = time.perf_counter()

latency = (end - start) / 100 * 1000

# Model size
size_mb = os.path.getsize(
    "models/mobilenetv2_leaf_best.pth"
) / (1024 * 1024)

print(f"CPU Latency: {latency:.2f} ms/image")
print(f"Model Size: {size_mb:.2f} MB")