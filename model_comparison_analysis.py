"""
Comprehensive Model Comparison and Analysis
Analyzes and compares all 4 models for pneumonia detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys

class ModelComparison:
    """Comprehensive model comparison and analysis."""
    
    def __init__(self):
        self.results = {}
        self.comparison_df = None
        
    def add_model_results(self, model_name, results):
        """Add model results to comparison."""
        self.results[model_name] = results
        
    def create_comparison_table(self):
        """Create comprehensive comparison table."""
        if not self.results:
            print("No results to compare. Please add model results first.")
            return None
            
        # Create comparison DataFrame
        comparison_data = []
        for model_name, model_results in self.results.items():
            if isinstance(model_results, dict):
                row = {'Model': model_name}
                for metric, value in model_results.items():
                    if isinstance(value, (int, float)):
                        row[metric] = value
                comparison_data.append(row)
        
        if comparison_data:
            self.comparison_df = pd.DataFrame(comparison_data)
            return self.comparison_df
        else:
            print("No valid results to compare.")
            return None
    
    def print_comparison_table(self):
        """Print formatted comparison table."""
        if self.comparison_df is None:
            self.create_comparison_table()
            
        if self.comparison_df is not None:
            print("\n" + "="*100)
            print("MODEL PERFORMANCE COMPARISON")
            print("="*100)
            
            # Format the dataframe for display
            display_df = self.comparison_df.round(4)
            print(display_df.to_string(index=False))
            
            # Find best model for each metric
            metrics = ['accuracy', 'precision', 'recall', 'f1-score']
            available_metrics = [m for m in metrics if m in self.comparison_df.columns]
            
            if available_metrics:
                print("\n" + "="*100)
                print("BEST PERFORMING MODELS")
                print("="*100)
                
                for metric in available_metrics:
                    if metric in self.comparison_df.columns:
                        best_model = self.comparison_df.loc[self.comparison_df[metric].idxmax()]
                        print(f"Best {metric.upper()}: {best_model['Model']} ({best_model[metric]:.4f})")
            
            print("="*100)
    
    def create_visualizations(self, save_dir="plots"):
        """Create comparison visualizations."""
        if self.comparison_df is None:
            print("No comparison data available.")
            return
            
        # Create plots directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. Accuracy Comparison
        if 'accuracy' in self.comparison_df.columns:
            sns.barplot(data=self.comparison_df, x='Model', y='accuracy', ax=axes[0, 0])
            axes[0, 0].set_title('Accuracy Comparison')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].set_ylabel('Accuracy')
        
        # 2. Precision vs Recall
        if 'precision' in self.comparison_df.columns and 'recall' in self.comparison_df.columns:
            sns.scatterplot(data=self.comparison_df, x='precision', y='recall', 
                           hue='Model', s=100, ax=axes[0, 1])
            axes[0, 1].set_title('Precision vs Recall')
            axes[0, 1].set_xlabel('Precision')
            axes[0, 1].set_ylabel('Recall')
            axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 3. F1-Score Comparison
        if 'f1-score' in self.comparison_df.columns:
            sns.barplot(data=self.comparison_df, x='Model', y='f1-score', ax=axes[1, 0])
            axes[1, 0].set_title('F1-Score Comparison')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].set_ylabel('F1-Score')
        
        # 4. ROC-AUC Comparison (if available)
        if 'roc-auc' in self.comparison_df.columns:
            sns.barplot(data=self.comparison_df, x='Model', y='roc-auc', ax=axes[1, 1])
            axes[1, 1].set_title('ROC-AUC Comparison')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].set_ylabel('ROC-AUC')
        else:
            # Alternative: Sensitivity vs Specificity
            if 'sensitivity' in self.comparison_df.columns and 'specificity' in self.comparison_df.columns:
                sns.scatterplot(data=self.comparison_df, x='sensitivity', y='specificity', 
                               hue='Model', s=100, ax=axes[1, 1])
                axes[1, 1].set_title('Sensitivity vs Specificity')
                axes[1, 1].set_xlabel('Sensitivity')
                axes[1, 1].set_ylabel('Specificity')
                axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                axes[1, 1].text(0.5, 0.5, 'No additional metrics available', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Additional Metrics')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/model_comparison_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Comparison plots saved to {save_dir}/model_comparison_analysis.png")
    
    def medical_interpretation(self):
        """Provide medical interpretation of results."""
        if self.comparison_df is None:
            print("No comparison data available.")
            return
            
        print("\n" + "="*100)
        print("MEDICAL INTERPRETATION")
        print("="*100)
        
        for _, row in self.comparison_df.iterrows():
            model_name = row['Model']
            print(f"\n{model_name}:")
            
            # Accuracy interpretation
            if 'accuracy' in row:
                acc = row['accuracy']
                if acc >= 0.90:
                    print(f"  ✅ EXCELLENT: {acc:.1%} accuracy - Exceeds medical standards")
                elif acc >= 0.85:
                    print(f"  ✅ VERY GOOD: {acc:.1%} accuracy - Meets medical standards")
                elif acc >= 0.75:
                    print(f"  ✅ GOOD: {acc:.1%} accuracy - Clinically useful")
                elif acc >= 0.65:
                    print(f"  ⚠️ FAIR: {acc:.1%} accuracy - Needs improvement")
                else:
                    print(f"  ❌ POOR: {acc:.1%} accuracy - Not suitable for clinical use")
            
            # Sensitivity and Specificity
            if 'sensitivity' in row and 'specificity' in row:
                sens = row['sensitivity']
                spec = row['specificity']
                print(f"  📊 Sensitivity: {sens:.1%} (detects {sens:.1%} of pneumonia cases)")
                print(f"  📊 Specificity: {spec:.1%} (correctly identifies {spec:.1%} of normal cases)")
            
            # Precision
            if 'precision' in row:
                prec = row['precision']
                print(f"  📊 Precision: {prec:.1%} (when predicting pneumonia, {prec:.1%} are correct)")
        
        print("\n" + "="*100)
        print("MEDICAL RECOMMENDATIONS")
        print("="*100)
        print("• Sensitivity > 80%: Good at detecting pneumonia cases")
        print("• Specificity > 80%: Good at identifying normal cases")
        print("• Precision > 80%: Low false positive rate")
        print("• F1-Score > 80%: Balanced performance")
        print("• Overall: Models with >85% accuracy are suitable for clinical screening")
    
    def save_results(self, filename=None):
        """Save comparison results to CSV."""
        if self.comparison_df is None:
            print("No comparison data to save.")
            return
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_comparison_analysis_{timestamp}.csv"
        
        self.comparison_df.to_csv(filename, index=False)
        print(f"📁 Results saved to: {filename}")
    
    def load_existing_results(self):
        """Load existing results from CSV files."""
        csv_files = [
            "model_comparison_results.csv",
            "accuracy_test_results.csv"
        ]
        
        loaded_results = {}
        
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                try:
                    df = pd.read_csv(csv_file)
                    print(f"📁 Loaded results from {csv_file}")
                    
                    # Process each row
                    for _, row in df.iterrows():
                        model_name = row.get('Model', f'Model_{len(loaded_results)}')
                        results = row.to_dict()
                        del results['Model']  # Remove model name from results
                        loaded_results[model_name] = results
                        
                except Exception as e:
                    print(f"❌ Error loading {csv_file}: {e}")
        
        # Add loaded results
        for model_name, results in loaded_results.items():
            self.add_model_results(model_name, results)
        
        return len(loaded_results) > 0

def run_comprehensive_analysis():
    """Run comprehensive model analysis."""
    print("="*100)
    print("🫁 PNEUMONIA DETECTION - COMPREHENSIVE MODEL ANALYSIS")
    print("="*100)
    
    # Initialize comparison
    comparison = ModelComparison()
    
    # Load existing results
    print("\n1. Loading Existing Results...")
    if comparison.load_existing_results():
        print("✅ Existing results loaded successfully!")
    else:
        print("⚠️ No existing results found.")
    
    # Create comparison table
    print("\n2. Creating Comparison Table...")
    comparison.create_comparison_table()
    
    # Print comparison
    print("\n3. Model Performance Comparison:")
    comparison.print_comparison_table()
    
    # Create visualizations
    print("\n4. Creating Visualizations...")
    comparison.create_visualizations()
    
    # Medical interpretation
    print("\n5. Medical Interpretation:")
    comparison.medical_interpretation()
    
    # Save results
    print("\n6. Saving Results...")
    comparison.save_results()
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETED!")
    print("="*100)
    
    return comparison

if __name__ == "__main__":
    try:
        comparison = run_comprehensive_analysis()
        print("\nComprehensive model analysis completed!")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
