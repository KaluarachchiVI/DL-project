"""
Simple Logistic Regression for Pneumonia Detection
Scikit-learn implementation
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

class SimpleLogisticRegression:
    """Simple Logistic Regression for pneumonia detection."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=100, random_state=random_state)
    
    def preprocess_data(self, image_paths, labels):
        """Preprocess images for logistic regression."""
        processed_images = []
        
        for image_path in image_paths:
            # Load and resize image
            from PIL import Image
            import cv2
            
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            
            # Flatten image
            flattened = image.flatten()
            processed_images.append(flattened)
        
        return np.array(processed_images), np.array(labels)
    
    def create_pipeline(self):
        """Create preprocessing and modeling pipeline."""
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=100, random_state=self.random_state)),
            ('classifier', LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            ))
        ])
    
    def train(self, X_train, y_train):
        """Train the model."""
        self.create_pipeline()
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        """Make predictions."""
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        """Get prediction probabilities."""
        return self.model.predict_proba(X_test)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model."""
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred
        }
    
    def save_model(self, filepath="results/logistic_model.pkl"):
        """Save the trained model."""
        joblib.dump(self.model, filepath)

def train_logistic_regression():
    """Train Logistic Regression model."""
    print("="*60)
    print("LOGISTIC REGRESSION TRAINING")
    print("="*60)
    
    # Import data preprocessing
    import sys
    sys.path.append('..')
    from utils.data_preprocessing import DataPreprocessor
    
    # Load data
    preprocessor = DataPreprocessor("../data", (224, 224), 32, 42)
    image_paths, labels = preprocessor.load_data_paths()
    X_train, X_val, y_train, y_val = preprocessor.create_train_val_split(image_paths, labels)
    
    # Preprocess images
    print("Preprocessing images...")
    lr = SimpleLogisticRegression()
    X_train_processed, y_train_processed = lr.preprocess_data(X_train, y_train)
    X_val_processed, y_val_processed = lr.preprocess_data(X_val, y_val)
    
    print(f"Training features shape: {X_train_processed.shape}")
    print(f"Validation features shape: {X_val_processed.shape}")
    
    # Train model
    print("\nTraining Logistic Regression...")
    lr.train(X_train_processed, y_train_processed)
    
    # Evaluate
    results = lr.evaluate(X_val_processed, y_val_processed)
    print(f"\nResults: Accuracy = {results['accuracy']:.4f}")
    
    # Save model
    lr.save_model()
    
    return lr, results

if __name__ == "__main__":
    train_logistic_regression()
