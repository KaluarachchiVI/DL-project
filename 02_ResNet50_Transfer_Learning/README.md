# Model 2: ResNet50 Transfer Learning (TensorFlow/Keras)

## Overview
Transfer Learning model using pre-trained ResNet50 architecture for chest X-ray pneumonia classification.

## Architecture
- **Base Model**: Pre-trained ResNet50 (ImageNet weights)
- **Custom Head**: Global Average Pooling + Dense layers
- **Fine-tuning**: Two-phase training (frozen + fine-tuned)

## Key Features
- Pre-trained ImageNet weights
- Transfer learning approach
- Fine-tuning capabilities
- Multiple base model options (ResNet50, DenseNet121, VGG16, EfficientNet)
- Advanced regularization (L2, Dropout, BatchNorm)

## Files
- `resnet_transfer.py` - Main model implementation
- `train_resnet.py` - Training script
- `evaluate_resnet.py` - Evaluation script
- `results/` - Model results and plots

## Performance
- **Accuracy**: 89.1%
- **Precision**: 89.1%
- **Recall**: 100%
- **F1-Score**: 94.3%

## Usage
```python
from resnet_transfer import TransferLearningModel

# Create model
transfer_model = TransferLearningModel()
transfer_model.build_model(base_model='resnet50')
transfer_model.compile_model()

# Train model
history = transfer_model.train(train_generator, val_generator, 
                              epochs=30, fine_tuning_epochs=10)
```

## Model Architecture
```
Input (224, 224, 3)
├── ResNet50 (Pre-trained, frozen initially)
├── GlobalAveragePooling2D()
├── BatchNormalization()
├── Dropout(0.5)
├── Dense(512, ReLU) + BatchNorm + Dropout(0.5)
├── Dense(256, ReLU) + BatchNorm + Dropout(0.25)
└── Dense(2, softmax)  # Output layer
```

## Training Phases
1. **Phase 1**: Train only classification head (frozen backbone)
2. **Phase 2**: Fine-tune last 10 layers with lower learning rate
