import argparse
import json
import os

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ---------------------------------
# DEVICE
# ---------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------
# ARGUMENTS
# ---------------------------------

parser = argparse.ArgumentParser(description="Leaf Disease Prediction")

parser.add_argument(
    "--image",
    required=True,
    help="Path to image"
)

parser.add_argument(
    "--model",
    default="models/resnet18_best.pth",
    help="Model checkpoint"
)

parser.add_argument(
    "--config",
    default="models/inference_config.json",
    help="Inference config"
)

args = parser.parse_args()

# ---------------------------------
# CHECK IMAGE
# ---------------------------------

if not os.path.exists(args.image):
    print("ERROR: Image not found.")
    exit(1)

# ---------------------------------
# LOAD CLASS NAMES
# ---------------------------------

with open("models/class_names.json", "r") as f:
    class_names = json.load(f)

NUM_CLASSES = len(class_names)

# ---------------------------------
# LOAD THRESHOLD
# ---------------------------------

threshold = 0.5

if os.path.exists(args.config):
    with open(args.config, "r") as f:
        config = json.load(f)
        threshold = config["healthy_vs_disease_threshold"]

# ---------------------------------
# LOAD MODEL
# ---------------------------------

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    NUM_CLASSES
)

checkpoint = torch.load(
    args.model,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()

# ---------------------------------
# IMAGE TRANSFORM
# ---------------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

image = Image.open(args.image).convert("RGB")

image = transform(image)

image = image.unsqueeze(0).to(device)

# ---------------------------------
# PREDICTION
# ---------------------------------

with torch.no_grad():

    outputs = model(image)

    probs = torch.softmax(outputs, dim=1)[0]

confidence, idx = torch.max(probs, dim=0)

predicted = class_names[idx.item()]

healthy_index = class_names.index("healthy")

disease_probability = 1 - probs[healthy_index].item()

predicted_class = predicted
is_diseased = predicted_class != "healthy"
# ---------------------------------
# RESULTS
# ---------------------------------

print("\n========== RESULT ==========")

print("Prediction :", predicted)

print("Confidence :", f"{confidence.item()*100:.2f}%")

print("Disease Probability :", f"{disease_probability:.4f}")

print("Threshold :", threshold)

print("Diseased :", is_diseased)

print("\nClass Probabilities")

for name, prob in zip(class_names, probs):

    print(f"{name:15} : {prob.item():.4f}")