"""
Logistic Regression Training Script
Trains the Logistic Regression model for pneumonia detection
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append('..')
from logistic_regression import LogisticRegressionClassifier
from utils.data_preprocessing import DataPreprocessor
from utils.evaluation import ModelEvaluator
from utils.visualization import Visualizer

def train_logistic_regression():
    """Train Logistic Regression model."""
    
    print("="*80)
    print("LOGISTIC REGRESSION TRAINING")
    print("="*80)
    print("Training Logistic Regression for Pneumonia Detection")
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
    
    # Preprocess images for sklearn
    print("\n2. Preprocessing Images...")
    print("Converting images to flattened features...")
    X_train_processed, y_train_processed = preprocessor.preprocess_for_sklearn(X_train, y_train)
    X_val_processed, y_val_processed = preprocessor.preprocess_for_sklearn(X_val, y_val)
    
    print(f"Training features shape: {X_train_processed.shape}")
    print(f"Validation features shape: {X_val_processed.shape}")
    
    # Initialize classifier
    print("\n3. Building Model...")
    lr_classifier = LogisticRegressionClassifier()
    
    # Train model
    print("\n4. Training Model...")
    lr_classifier.train(
        X_train_processed, 
        y_train_processed, 
        use_pca=True, 
        n_components=100,
        grid_search=True,
        cv=5
    )
    
    # Cross-validation
    print("\n5. Cross-Validation...")
    cv_scores = lr_classifier.cross_validate(X_train_processed, y_train_processed, cv=5)
    
    # Evaluate model
    print("\n6. Evaluating Model...")
    results = lr_classifier.evaluate(X_val_processed, y_val_processed)
    
    # Feature importance
    print("\n7. Feature Importance Analysis...")
    importance_df = lr_classifier.get_feature_importance()
    print(f"Top 10 most important features:")
    print(importance_df.head(10))
    
    # Plot feature importance
    lr_classifier.plot_feature_importance(top_n=20)
    
    # Save model
    print("\n8. Saving Model...")
    lr_classifier.save_model("results/logistic_regression_model.pkl")
    
    # Print results
    print("\n" + "="*80)
    print("TRAINING COMPLETED!")
    print("="*80)
    print(f"Final Results:")
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
    
    print(f"\nCross-validation F1 scores: {cv_scores}")
    print(f"Mean F1 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return lr_classifier, results

if __name__ == "__main__":
    try:
        model, results = train_logistic_regression()
        print("\nLogistic Regression training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
