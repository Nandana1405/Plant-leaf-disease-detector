# Plant Leaf Disease Detector

## Project Description
This project uses Deep Learning with PyTorch to detect diseases in plant leaves from images.

## Project Structure

- data/
  - raw/
  - processed/
- models/
- notebooks/
- src/

## Environment
- Python 3.10
- PyTorch
- TorchVision
- Pillow
- NumPy
- Matplotlib

## Environment Verification

- PyTorch Version: 2.13.0+cpu
- TorchVision Version: 0.28.0+cpu
- CUDA Available: False
- Device: CPU
- Scikit-learn

## Day 6 - Validation, Early Stopping & Loss Curves

- Implemented a validation loop using `model.eval()` and `torch.no_grad()`.
- Trained the CNN using both training and validation datasets.
- Applied early stopping to prevent overfitting.
- Saved the best model as `models/leaf_cnn_best.pth`.
- Generated and saved the training vs validation loss curve as `reports/training_curves.png`.

### Results
- Best Validation Accuracy: 79.63%
- Training completed successfully.
 
 ## Data Augmentation

Training data augmentation was applied using torchvision transforms.

The following label-safe transformations were used:
- RandomResizedCrop: changes crop and scale while preserving the leaf disease class.
- RandomHorizontalFlip: changes leaf orientation without changing the disease label.
- RandomRotation: simulates leaves captured at different camera angles.
- ColorJitter: simulates moderate variations in lighting and color conditions.

Validation images were not randomly augmented. They were only resized, converted to tensors, and normalized to ensure stable validation metrics.

A short 5-epoch retraining was performed using the augmented training dataset. The best validation accuracy obtained was approximately 77.38%.

## Class Balancing Results

Class balancing was implemented using weighted CrossEntropyLoss.

The dataset was already nearly balanced:
- Healthy: 1000 images
- Early Blight: 1000 images
- Late Blight: 999 images
- Leaf Mold: 1000 images

### Before Balancing
Baseline validation accuracy: 72.88%

### After Balancing
Best validation accuracy: 80.63%
Best validation loss: 0.5252
Best epoch: 5

Per-class recall:
- Healthy: 97.33%
- Early Blight: 77.05%
- Late Blight: 69.39%
- Leaf Mold: 76.02%

Weighted loss was used to account for differences in class frequency.
The balanced model achieved improved overall validation accuracy compared
with the previous baseline.