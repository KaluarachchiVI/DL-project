# Model 1: Custom CNN (TensorFlow/Keras)

## Overview
Custom Convolutional Neural Network implemented from scratch using TensorFlow/Keras for chest X-ray pneumonia classification.

## Architecture Options
- **Standard CNN**: 4 convolutional blocks (32→64→128→256 filters)
- **Deep CNN**: 5 convolutional blocks with double layers
- **Lightweight CNN**: 3 convolutional blocks (16→32→64 filters)

## Key Features
- Batch Normalization
- Dropout regularization (0.25-0.5)
- Global Average Pooling
- Multiple architecture options
- Early stopping and learning rate reduction

## Files
- `custom_cnn.py` - Main model implementation
- `train_custom_cnn.py` - Training script
- `evaluate_custom_cnn.py` - Evaluation script
- `results/` - Model results and plots

## Performance
- **Accuracy**: 89.1%
- **Precision**: 89.1%
- **Recall**: 100%
- **F1-Score**: 94.3%

## Usage
```python
from custom_cnn import CustomCNN

# Create model
cnn = CustomCNN()
cnn.build_model(architecture='standard')
cnn.compile_model()

# Train model
history = cnn.train(train_generator, val_generator, epochs=50)
```

## Model Architecture
```
Input (224, 224, 3)
├── Conv2D(32) + BatchNorm + ReLU + MaxPool + Dropout(0.25)
├── Conv2D(64) + BatchNorm + ReLU + MaxPool + Dropout(0.25)
├── Conv2D(128) + BatchNorm + ReLU + MaxPool + Dropout(0.25)
├── Conv2D(256) + BatchNorm + ReLU + MaxPool + Dropout(0.25)
├── GlobalAveragePooling2D()
├── Dense(512) + BatchNorm + ReLU + Dropout(0.5)
├── Dense(256) + BatchNorm + ReLU + Dropout(0.5)
└── Dense(2, softmax)  # Output layer
```
