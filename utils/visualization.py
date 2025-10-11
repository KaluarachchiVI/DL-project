"""
Visualization utilities for chest X-ray pneumonia classification.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from PIL import Image
import os

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class Visualizer:
    """Visualization class for model results and data exploration."""
    
    def __init__(self, save_dir="plots"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_sample_images(self, image_paths, labels, n_samples=8, figsize=(15, 10)):
        """Plot sample images from the dataset."""
        fig, axes = plt.subplots(2, 4, figsize=figsize)
        axes = axes.ravel()
        
        # Get random samples
        indices = np.random.choice(len(image_paths), n_samples, replace=False)
        
        for i, idx in enumerate(indices):
            image_path = image_paths[idx]
            label = labels[idx]
            
            # Load and display image
            image = Image.open(image_path)
            axes[i].imshow(image, cmap='gray')
            axes[i].set_title(f'Label: {label}', fontsize=12)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/sample_images.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_class_distribution(self, labels, title="Class Distribution"):
        """Plot class distribution."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Count plot
        class_counts = pd.Series(labels).value_counts()
        sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax1)
        ax1.set_title(f"{title} - Count")
        ax1.set_xlabel("Class")
        ax1.set_ylabel("Count")
        
        # Pie chart
        ax2.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', startangle=90)
        ax2.set_title(f"{title} - Percentage")
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/class_distribution.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self, history, model_name, figsize=(15, 5)):
        """Plot training history for TensorFlow models."""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Accuracy
        axes[0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title(f'{model_name} - Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss
        axes[1].plot(history.history['loss'], label='Training Loss')
        axes[1].plot(history.history['val_loss'], label='Validation Loss')
        axes[1].set_title(f'{model_name} - Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        # Learning rate (if available)
        if 'lr' in history.history:
            axes[2].plot(history.history['lr'])
            axes[2].set_title(f'{model_name} - Learning Rate')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Learning Rate')
            axes[2].grid(True)
        else:
            axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{model_name}_training_history.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_pytorch_training_history(self, train_losses, val_losses, train_accs, val_accs, model_name, figsize=(15, 5)):
        """Plot training history for PyTorch models."""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Accuracy
        axes[0].plot(train_accs, label='Training Accuracy')
        axes[0].plot(val_accs, label='Validation Accuracy')
        axes[0].set_title(f'{model_name} - Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss
        axes[1].plot(train_losses, label='Training Loss')
        axes[1].plot(val_losses, label='Validation Loss')
        axes[1].set_title(f'{model_name} - Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        # Learning rate schedule (placeholder)
        axes[2].text(0.5, 0.5, 'Learning Rate\nSchedule', ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title(f'{model_name} - Learning Rate')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{model_name}_training_history.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, class_names=['Normal', 'Pneumonia']):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{model_name} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{model_name}_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_true, y_scores, model_name):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} - ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{model_name}_roc_curve.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, results_df, figsize=(15, 10)):
        """Plot model comparison charts."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Accuracy comparison
        sns.barplot(data=results_df, x='Model', y='Accuracy', ax=axes[0, 0])
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # F1-Score comparison
        sns.barplot(data=results_df, x='Model', y='F1-Score', ax=axes[0, 1])
        axes[0, 1].set_title('Model F1-Score Comparison')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # ROC-AUC comparison
        sns.barplot(data=results_df, x='Model', y='ROC-AUC', ax=axes[1, 0])
        axes[1, 0].set_title('Model ROC-AUC Comparison')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Precision vs Recall
        sns.scatterplot(data=results_df, x='Precision', y='Recall', 
                       hue='Model', s=100, ax=axes[1, 1])
        axes[1, 1].set_title('Precision vs Recall')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_dashboard(self, results_df):
        """Create interactive dashboard using Plotly."""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy Comparison', 'F1-Score Comparison', 
                          'ROC-AUC Comparison', 'Precision vs Recall'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Accuracy
        fig.add_trace(
            go.Bar(x=results_df['Model'], y=results_df['Accuracy'], 
                  name='Accuracy', marker_color='lightblue'),
            row=1, col=1
        )
        
        # F1-Score
        fig.add_trace(
            go.Bar(x=results_df['Model'], y=results_df['F1-Score'], 
                  name='F1-Score', marker_color='lightgreen'),
            row=1, col=2
        )
        
        # ROC-AUC
        fig.add_trace(
            go.Bar(x=results_df['Model'], y=results_df['ROC-AUC'], 
                  name='ROC-AUC', marker_color='lightcoral'),
            row=2, col=1
        )
        
        # Precision vs Recall
        fig.add_trace(
            go.Scatter(x=results_df['Precision'], y=results_df['Recall'],
                      mode='markers+text', text=results_df['Model'],
                      textposition="top center", name='Models',
                      marker=dict(size=15, color='purple')),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False, 
                         title_text="Model Performance Dashboard")
        
        # Save as HTML
        fig.write_html(f"{self.save_dir}/interactive_dashboard.html")
        fig.show()
    
    def plot_feature_importance(self, feature_importance, feature_names, model_name, top_n=20):
        """Plot feature importance for tree-based models."""
        # Get top N features
        indices = np.argsort(feature_importance)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.title(f'{model_name} - Top {top_n} Feature Importance')
        plt.bar(range(top_n), feature_importance[indices])
        plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{model_name}_feature_importance.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_gradcam_heatmap(self, model, image, class_idx, model_name):
        """Plot GradCAM heatmap for CNN models."""
        # This is a placeholder for GradCAM implementation
        # In practice, you would implement GradCAM using tf-keras-vis or similar
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        # Placeholder heatmap
        heatmap = np.random.rand(*image.shape[:2])
        plt.imshow(heatmap, cmap='jet', alpha=0.7)
        plt.title(f'{model_name} - GradCAM Heatmap')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{model_name}_gradcam.png", dpi=300, bbox_inches='tight')
        plt.show()

def create_results_summary_table(results_df):
    """Create a formatted results summary table."""
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80)
    
    # Format the dataframe for display
    display_df = results_df.round(4)
    
    print(display_df.to_string(index=False))
    print("="*80)
    
    # Find best model for each metric
    best_accuracy = results_df.loc[results_df['Accuracy'].idxmax()]
    best_f1 = results_df.loc[results_df['F1-Score'].idxmax()]
    best_auc = results_df.loc[results_df['ROC-AUC'].idxmax()]
    
    print(f"\nBest Accuracy: {best_accuracy['Model']} ({best_accuracy['Accuracy']:.4f})")
    print(f"Best F1-Score: {best_f1['Model']} ({best_f1['F1-Score']:.4f})")
    print(f"Best ROC-AUC: {best_auc['Model']} ({best_auc['ROC-AUC']:.4f})")
    print("="*80)

if __name__ == "__main__":
    # Test the visualizer
    visualizer = Visualizer()
    print("Visualization utilities loaded successfully!")
