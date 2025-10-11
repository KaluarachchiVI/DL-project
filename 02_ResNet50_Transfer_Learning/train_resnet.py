"""
ResNet50 Transfer Learning Training Script
Trains the ResNet50 transfer learning model for pneumonia detection
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append('..')
from resnet_transfer import TransferLearningModel
from utils.data_preprocessing import DataPreprocessor
from utils.evaluation import ModelEvaluator
from utils.visualization import Visualizer

def train_resnet_transfer():
    """Train ResNet50 Transfer Learning model."""
    
    print("="*80)
    print("RESNET50 TRANSFER LEARNING TRAINING")
    print("="*80)
    print("Training ResNet50 Transfer Learning for Pneumonia Detection")
    print("="*80)
    
    # Initialize components
    preprocessor = DataPreprocessor("../data", (224, 224), 32, 42)
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
    
    # Create data generators
    print("\n2. Creating Data Generators...")
    train_generator, val_generator = preprocessor.get_tensorflow_generators(
        X_train, X_val, y_train, y_val, augmentation=True
    )
    
    # Initialize and build model
    print("\n3. Building Model...")
    transfer_model = TransferLearningModel()
    transfer_model.build_model(base_model='resnet50', trainable_layers=None)
    transfer_model.compile_model(learning_rate=0.001)
    
    # Print model summary
    print("\nModel Architecture:")
    transfer_model.get_model_summary()
    
    # Train model
    print("\n4. Training Model...")
    history = transfer_model.train(
        train_generator, 
        val_generator, 
        epochs=30,
        fine_tuning_epochs=10,
        callbacks=transfer_model._get_default_callbacks()
    )
    
    # Plot training history
    print("\n5. Plotting Training History...")
    transfer_model.plot_training_history()
    
    # Evaluate model
    print("\n6. Evaluating Model...")
    test_generator = preprocessor.get_tensorflow_generators(
        X_train, X_val, y_train, y_val, augmentation=False
    )[1]  # Use validation generator as test
    
    results = transfer_model.evaluate(test_generator)
    
    # Save model
    print("\n7. Saving Model...")
    transfer_model.save_model("results/resnet_transfer_best.h5")
    
    # Print results
    print("\n" + "="*80)
    print("TRAINING COMPLETED!")
    print("="*80)
    print(f"Final Results:")
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
    
    return transfer_model, results, history

if __name__ == "__main__":
    try:
        model, results, history = train_resnet_transfer()
        print("\nResNet50 Transfer Learning training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
