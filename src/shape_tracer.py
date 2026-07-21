import torch
import torch.nn as nn

def trace_shapes(x, layers):
    for layer in layers:
        x = layer(x)
        print(f"{layer.__class__.__name__:12s} -> {tuple(x.shape)}")

x = torch.randn(1, 3, 224, 224)  # one leaf batch
layers = [
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),           # 112
    nn.Conv2d(32, 64, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),           # 56
    nn.Conv2d(64, 128, 3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),   # 1x1 spatial
]
trace_shapes(x, layers)
# Expect final torch.Size([1, 128, 1, 1]) before flatten + linear head