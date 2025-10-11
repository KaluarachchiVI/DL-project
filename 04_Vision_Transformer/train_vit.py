"""
Vision Transformer Training Script
Trains the Vision Transformer model for pneumonia detection
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append('..')
from vision_transformer import VisionTransformerClassifier, train_vision_transformer
from utils.data_preprocessing import DataPreprocessor, ChestXRayDataset
from utils.evaluation import ModelEvaluator
from utils.visualization import Visualizer

def train_vision_transformer_model():
    """Train Vision Transformer model."""
    
    print("="*80)
    print("VISION TRANSFORMER TRAINING")
    print("="*80)
    print("Training Vision Transformer for Pneumonia Detection")
    print("="*80)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize components
    preprocessor = DataPreprocessor("../data", (224, 224), 8, 42)  # Smaller batch size for ViT
    evaluator = ModelEvaluator()
    visualizer = Visualizer("results")
    
    # Load data paths
    print("\n1. Loading Data...")
    image_paths, labels = preprocessor.load_data_paths()
    print(f"Total images: {len(image_paths)}")
    
    # Create train/val split
    X_train, X_val, y_train, y_val = preprocessor.create_train_val_split(image_paths, labels)
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Create PyTorch data loaders
    print("\n2. Creating Data Loaders...")
    train_loader, val_loader = preprocessor.get_pytorch_dataloaders(
        X_train, X_val, y_train, y_val, num_workers=2
    )
    
    # Create test loader (using validation data for now)
    test_dataset = ChestXRayDataset(X_val, y_val, preprocessor.get_pytorch_transforms()[1])
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    # Train model
    print("\n3. Training Vision Transformer...")
    try:
        # Use pre-trained ViT
        vit, results, trainer = train_vision_transformer(
            train_loader, 
            val_loader, 
            test_loader,
            model_type='pretrained', 
            epochs=5  # Reduced for demo
        )
        
        print("\n4. Training Completed Successfully!")
        
        # Save model
        print("\n5. Saving Model...")
        vit.save_model("results/vision_transformer_model")
        
        # Print results
        print("\n" + "="*80)
        print("TRAINING COMPLETED!")
        print("="*80)
        print(f"Final Results:")
        for metric, value in results.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
        
        return vit, results, trainer
        
    except Exception as e:
        print(f"\nError with pre-trained ViT: {e}")
        print("Trying custom ViT implementation...")
        
        try:
            # Use custom ViT
            model, results, trainer = train_vision_transformer(
                train_loader, 
                val_loader, 
                test_loader,
                model_type='custom', 
                epochs=10
            )
            
            print("\nCustom ViT training completed successfully!")
            return model, results, trainer
            
        except Exception as e2:
            print(f"Error with custom ViT: {e2}")
            print("Vision Transformer training failed. Please check your environment.")
            return None, None, None

if __name__ == "__main__":
    try:
        model, results, trainer = train_vision_transformer_model()
        if model is not None:
            print("\nVision Transformer training completed successfully!")
        else:
            print("\nVision Transformer training failed.")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
