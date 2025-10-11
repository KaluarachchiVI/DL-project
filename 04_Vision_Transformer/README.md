# Model 4: Vision Transformer (PyTorch)

## Overview
State-of-the-art Vision Transformer (ViT) model for chest X-ray pneumonia classification using PyTorch and Hugging Face Transformers.

## Architecture Options
- **Pre-trained ViT**: google/vit-base-patch16-224 (Hugging Face)
- **Custom ViT**: Custom implementation with configurable parameters

## Key Features
- Pre-trained transformer weights
- Patch-based image processing
- Multi-head self-attention
- Position embeddings
- Hugging Face integration
- Custom ViT implementation option

## Files
- `vision_transformer.py` - Main model implementation
- `train_vit.py` - Training script
- `evaluate_vit.py` - Evaluation script
- `results/` - Model results and plots

## Performance
- **Architecture**: Transformer-based
- **Input**: 224x224 patches
- **Attention**: Multi-head self-attention
- **Parameters**: ~86M (ViT-Base)

## Usage
```python
from vision_transformer import VisionTransformerClassifier

# Create model
vit = VisionTransformerClassifier(device=device)
vit.build_model()

# Train model
trainer = vit.train(train_loader, val_loader, epochs=10)
```

## Model Architecture
```
Input Image (224, 224, 3)
├── Patch Embedding (16x16 patches)
├── Position Embedding
├── Class Token
├── Transformer Encoder (12 layers)
│   ├── Multi-Head Self-Attention
│   ├── Layer Normalization
│   ├── MLP
│   └── Residual Connections
├── Layer Normalization
└── Classification Head (2 classes)
```

## Training Features
- **Optimizer**: AdamW with weight decay
- **Learning Rate**: 2e-5 (pre-trained) / 1e-4 (custom)
- **Scheduler**: Cosine annealing
- **Batch Size**: 8 (GPU memory optimized)
- **Warmup Steps**: 500
