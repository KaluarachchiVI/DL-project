"""
Logistic Regression model for chest X-ray pneumonia classification.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
import os
from utils.evaluation import ModelEvaluator
from utils.visualization import Visualizer

class LogisticRegressionClassifier:
    """Logistic Regression classifier for chest X-ray images."""
    
    def __init__(self, random_state=42, max_iter=1000):
        self.random_state = random_state
        self.max_iter = max_iter
        self.model = None
        self.scaler = None
        self.pca = None
        self.pipeline = None
        self.evaluator = ModelEvaluator()
        self.visualizer = Visualizer()
        
    def create_pipeline(self, use_pca=True, n_components=100):
        """Create preprocessing and modeling pipeline."""
        steps = []
        
        # Add scaler
        steps.append(('scaler', StandardScaler()))
        
        # Add PCA if requested
        if use_pca:
            steps.append(('pca', PCA(n_components=n_components, random_state=self.random_state)))
        
        # Add logistic regression
        steps.append(('classifier', LogisticRegression(
            random_state=self.random_state,
            max_iter=self.max_iter,
            class_weight='balanced'
        )))
        
        self.pipeline = Pipeline(steps)
        return self.pipeline
    
    def train(self, X_train, y_train, use_pca=True, n_components=100, 
              grid_search=True, cv=5):
        """Train the logistic regression model."""
        
        print("Training Logistic Regression model...")
        
        # Create pipeline
        self.create_pipeline(use_pca=use_pca, n_components=n_components)
        
        if grid_search:
            # Define parameter grid
            param_grid = {
                'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__solver': ['liblinear', 'saga']
            }
            
            # Perform grid search
            grid_search = GridSearchCV(
                self.pipeline, 
                param_grid, 
                cv=cv, 
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            self.pipeline = grid_search.best_estimator_
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        else:
            # Train without grid search
            self.pipeline.fit(X_train, y_train)
        
        # Store components for later use
        self.scaler = self.pipeline.named_steps['scaler']
        if use_pca:
            self.pca = self.pipeline.named_steps['pca']
        self.model = self.pipeline.named_steps['classifier']
        
        print("Logistic Regression training completed!")
        return self.pipeline
    
    def predict(self, X_test):
        """Make predictions on test data."""
        if self.pipeline is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        return self.pipeline.predict(X_test)
    
    def predict_proba(self, X_test):
        """Get prediction probabilities."""
        if self.pipeline is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        return self.pipeline.predict_proba(X_test)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model."""
        if self.pipeline is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Get predictions
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        # Evaluate
        results = self.evaluator.evaluate_binary_classification(
            y_test, y_pred, y_pred_proba, "Logistic Regression"
        )
        
        return results
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation."""
        if self.pipeline is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        cv_scores = cross_val_score(self.pipeline, X, y, cv=cv, scoring='f1')
        
        print(f"Cross-validation F1 scores: {cv_scores}")
        print(f"Mean F1 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_scores
    
    def get_feature_importance(self, feature_names=None):
        """Get feature importance (coefficients)."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Get coefficients
        coefficients = self.model.coef_[0]
        
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(coefficients))]
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)
        
        return importance_df
    
    def plot_feature_importance(self, top_n=20):
        """Plot feature importance."""
        importance_df = self.get_feature_importance()
        top_features = importance_df.head(top_n)
        
        self.visualizer.plot_feature_importance(
            top_features['coefficient'].values,
            top_features['feature'].values,
            "Logistic Regression",
            top_n
        )
    
    def save_model(self, filepath="models/saved/logistic_regression_model.pkl"):
        """Save the trained model."""
        if self.pipeline is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.pipeline, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath="models/saved/logistic_regression_model.pkl"):
        """Load a trained model."""
        self.pipeline = joblib.load(filepath)
        self.scaler = self.pipeline.named_steps['scaler']
        if 'pca' in self.pipeline.named_steps:
            self.pca = self.pipeline.named_steps['pca']
        self.model = self.pipeline.named_steps['classifier']
        print(f"Model loaded from {filepath}")

def train_logistic_regression(X_train, X_test, y_train, y_test, 
                             use_pca=True, n_components=100):
    """Train and evaluate logistic regression model."""
    
    # Initialize classifier
    lr_classifier = LogisticRegressionClassifier()
    
    # Train model
    lr_classifier.train(X_train, y_train, use_pca=use_pca, n_components=n_components)
    
    # Evaluate model
    results = lr_classifier.evaluate(X_test, y_test)
    
    # Cross-validation
    cv_scores = lr_classifier.cross_validate(X_train, y_train)
    
    # Save model
    lr_classifier.save_model()
    
    return lr_classifier, results

if __name__ == "__main__":
    # Test the logistic regression classifier
    print("Logistic Regression model implementation loaded successfully!")
