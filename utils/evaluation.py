"""
Evaluation utilities for chest X-ray pneumonia classification.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import time
import torch
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

class ModelEvaluator:
    """Comprehensive model evaluation class."""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_binary_classification(self, y_true, y_pred, y_scores=None, model_name="Model"):
        """Evaluate binary classification model."""
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
        
        # ROC-AUC if probabilities available
        roc_auc = None
        if y_scores is not None:
            roc_auc = roc_auc_score(y_true, y_scores)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Store results
        results = {
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'Specificity': specificity,
            'Sensitivity': sensitivity,
            'True Positives': tp,
            'True Negatives': tn,
            'False Positives': fp,
            'False Negatives': fn
        }
        
        self.results[model_name] = results
        return results
    
    def evaluate_tensorflow_model(self, model, test_generator, model_name="TensorFlow Model"):
        """Evaluate TensorFlow model."""
        # Get predictions
        y_pred_proba = model.predict(test_generator)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Get true labels
        y_true = test_generator.classes
        
        # Evaluate
        results = self.evaluate_binary_classification(
            y_true, y_pred, y_pred_proba.flatten(), model_name
        )
        
        return results
    
    def evaluate_pytorch_model(self, model, test_loader, device, model_name="PyTorch Model"):
        """Evaluate PyTorch model."""
        model.eval()
        y_pred_list = []
        y_true_list = []
        y_scores_list = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                y_pred_list.extend(predictions.cpu().numpy())
                y_true_list.extend(labels.cpu().numpy())
                y_scores_list.extend(probabilities[:, 1].cpu().numpy())
        
        # Evaluate
        results = self.evaluate_binary_classification(
            y_true_list, y_pred_list, y_scores_list, model_name
        )
        
        return results
    
    def evaluate_sklearn_model(self, model, X_test, y_test, model_name="Sklearn Model"):
        """Evaluate scikit-learn model."""
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Get probabilities if available
        y_scores = None
        if hasattr(model, 'predict_proba'):
            y_scores = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        results = self.evaluate_binary_classification(
            y_test, y_pred, y_scores, model_name
        )
        
        return results
    
    def cross_validate_model(self, model, X, y, cv=5, scoring='accuracy'):
        """Perform cross-validation."""
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        return {
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'scores': cv_scores
        }
    
    def get_classification_report(self, y_true, y_pred, target_names=['Normal', 'Pneumonia']):
        """Get detailed classification report."""
        return classification_report(y_true, y_pred, target_names=target_names)
    
    def get_confusion_matrix_details(self, y_true, y_pred):
        """Get detailed confusion matrix analysis."""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        return {
            'confusion_matrix': cm,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp,
            'total_samples': len(y_true)
        }
    
    def calculate_confidence_intervals(self, scores, confidence=0.95):
        """Calculate confidence intervals for metrics."""
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        n = len(scores)
        
        # Calculate confidence interval
        alpha = 1 - confidence
        t_value = 1.96  # Approximate for 95% confidence
        margin_error = t_value * (std_score / np.sqrt(n))
        
        return {
            'mean': mean_score,
            'std': std_score,
            'lower_bound': mean_score - margin_error,
            'upper_bound': mean_score + margin_error,
            'confidence_level': confidence
        }
    
    def compare_models(self, results_dict):
        """Compare multiple models and return comparison dataframe."""
        comparison_df = pd.DataFrame(list(results_dict.values()))
        
        # Sort by accuracy
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        return comparison_df
    
    def get_best_model(self, metric='Accuracy'):
        """Get the best model based on specified metric."""
        if not self.results:
            return None
        
        comparison_df = self.compare_models(self.results)
        best_model = comparison_df.iloc[0]
        
        return {
            'model_name': best_model['Model'],
            'metric': metric,
            'score': best_model[metric],
            'all_metrics': best_model.to_dict()
        }
    
    def save_results(self, filename='model_results.csv'):
        """Save results to CSV file."""
        if not self.results:
            print("No results to save.")
            return
        
        comparison_df = self.compare_models(self.results)
        comparison_df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

class TrainingMonitor:
    """Monitor training progress and metrics."""
    
    def __init__(self):
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': []
        }
    
    def update(self, train_loss, val_loss, train_acc, val_acc, lr=None):
        """Update metrics history."""
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['val_loss'].append(val_loss)
        self.metrics_history['train_acc'].append(train_acc)
        self.metrics_history['val_acc'].append(val_acc)
        if lr is not None:
            self.metrics_history['learning_rate'].append(lr)
    
    def get_best_epoch(self, metric='val_acc'):
        """Get the epoch with the best metric."""
        if metric not in self.metrics_history:
            return None
        
        best_idx = np.argmax(self.metrics_history[metric])
        return {
            'epoch': best_idx + 1,
            'value': self.metrics_history[metric][best_idx]
        }
    
    def detect_overfitting(self, patience=5):
        """Detect overfitting based on validation loss."""
        if len(self.metrics_history['val_loss']) < patience:
            return False
        
        val_losses = self.metrics_history['val_loss']
        best_val_loss = min(val_losses)
        best_epoch = val_losses.index(best_val_loss)
        
        # Check if validation loss has been increasing for 'patience' epochs
        if len(val_losses) - best_epoch > patience:
            return True
        
        return False

def calculate_model_complexity(model):
    """Calculate model complexity metrics."""
    if hasattr(model, 'count_params'):  # TensorFlow/Keras model
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': non_trainable_params
        }
    
    elif hasattr(model, 'parameters'):  # PyTorch model
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': non_trainable_params
        }
    
    else:  # Scikit-learn model
        return {
            'model_type': type(model).__name__,
            'parameters': model.get_params() if hasattr(model, 'get_params') else 'N/A'
        }

def benchmark_model_performance(model, X_test, y_test, model_name="Model"):
    """Benchmark model performance including inference time."""
    start_time = time.time()
    
    # Make predictions
    if hasattr(model, 'predict'):
        y_pred = model.predict(X_test)
    else:
        # For PyTorch models, you'd need to implement prediction logic
        y_pred = None
    
    inference_time = time.time() - start_time
    
    # Calculate metrics
    if y_pred is not None:
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='binary')
    else:
        accuracy = f1 = None
    
    return {
        'model_name': model_name,
        'inference_time': inference_time,
        'samples_per_second': len(X_test) / inference_time if inference_time > 0 else 0,
        'accuracy': accuracy,
        'f1_score': f1
    }

if __name__ == "__main__":
    # Test the evaluator
    evaluator = ModelEvaluator()
    print("Evaluation utilities loaded successfully!")
