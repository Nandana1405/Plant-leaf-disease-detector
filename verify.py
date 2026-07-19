import torch
import torchvision

print(f"PyTorch Version: {torch.__version__}")
print(f"TorchVision Version: {torchvision.__version__}")

print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

x = torch.randn(2, 3, 224, 224)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = x.to(device)

print(f"Tensor Device: {x.device}")