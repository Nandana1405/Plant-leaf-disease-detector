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
