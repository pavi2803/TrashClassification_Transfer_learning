# Trash Classification Using Transfer Learning with PyTorch

## Project Overview

This project aims to build an image classification model to identify types of trash (e.g., cardboard, food organics, glass, metal, etc.) using transfer learning with pre-trained deep learning architectures such as ResNet, EfficientNet, and VGG.

---

## Dataset

- Dataset directory: `../data/PDF Project/RealWaste/`
- Classes (labels):
    
    ```
    bash
    CopyEdit
    ['1-Cardboard', '2-Food Organics', '3-Glass', '4-Metal',
     '5-Miscellaneous Trash', '6-Paper', '7-Plastic', '8-Textile Trash', '9-Vegetation']
    
    ```
    
- Images organized by class folders.
- Data split:
    - 80% for training
    - 20% for validation/testing

---

## Data Preprocessing & Augmentation

- **Training transforms:**
    - Random resized crop (224x224)
    - Random horizontal flip
    - Random rotation (±15°)
    - Color jitter (brightness & contrast)
    - Random affine translation (±10%)
    - Conversion to tensor
- **Validation/test transforms:**
    - Resize to 224x224
    - Conversion to tensor

---

## Model Architectures Compared

- ResNet50, ResNet100
- EfficientNetB0
- VGG16

All initialized with pretrained weights on ImageNet.

---

## Model Setup (Example: ResNet50)

- Load pretrained model.
- Freeze convolutional base layers to prevent updates.
- Replace the final fully connected layer with a custom classification head for the 9 classes.
- Custom head includes batch normalization, dropout (0.2), and linear layer followed by log softmax activation.

---

## Training Details

- Loss function: Negative Log Likelihood Loss (`nn.NLLLoss`) since model outputs log probabilities.
- Optimizer: Adam optimizer with learning rate = 0.001 and weight decay = 1e-4 applied only to the classifier head.
- Early stopping with patience of 10 epochs based on validation loss.
- Model checkpoints saved when validation loss improves.
- Batch size: 5 (can be adjusted based on hardware).
- Number of epochs: up to 100.

---

## Evaluation & Visualization

- Track and plot training and validation loss per epoch.
- Use early stopping to avoid overfitting.
- Visualize sample images with their predicted classes.
