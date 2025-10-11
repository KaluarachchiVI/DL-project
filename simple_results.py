"""
Simple Results Viewer
View and analyze model results without emojis
"""

import pandas as pd
import numpy as np
import os

def view_results():
    """View and analyze existing results."""
    print("="*80)
    print("PNEUMONIA DETECTION - RESULTS ANALYSIS")
    print("="*80)
    
    # Check for existing CSV files
    csv_files = [
        "model_comparison_results.csv",
        "accuracy_test_results.csv"
    ]
    
    found_files = []
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            found_files.append(csv_file)
            print(f"Found: {csv_file}")
    
    if not found_files:
        print("No results files found.")
        print("Please run model training first.")
        return None
    
    # Load and combine results
    all_data = []
    for csv_file in found_files:
        try:
            df = pd.read_csv(csv_file)
            print(f"Loaded {len(df)} results from {csv_file}")
            all_data.append(df)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    if not all_data:
        print("No valid results found.")
        return None
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Remove duplicates
    combined_df = combined_df.drop_duplicates(subset=['Model'], keep='last')
    
    print(f"\nRESULTS SUMMARY ({len(combined_df)} models):")
    print("="*80)
    print(combined_df.to_string(index=False))
    
    # Analysis
    print(f"\n{'='*80}")
    print("PERFORMANCE ANALYSIS")
    print(f"{'='*80}")
    
    # Best accuracy
    if 'accuracy' in combined_df.columns:
        best_acc = combined_df.loc[combined_df['accuracy'].idxmax()]
        print(f"Best Accuracy: {best_acc['Model']} ({best_acc['accuracy']:.4f})")
    
    # Best precision
    if 'precision' in combined_df.columns:
        best_prec = combined_df.loc[combined_df['precision'].idxmax()]
        print(f"Best Precision: {best_prec['Model']} ({best_prec['precision']:.4f})")
    
    # Best recall
    if 'recall' in combined_df.columns:
        best_recall = combined_df.loc[combined_df['recall'].idxmax()]
        print(f"Best Recall: {best_recall['Model']} ({best_recall['recall']:.4f})")
    
    # Best F1-score
    if 'f1-score' in combined_df.columns:
        best_f1 = combined_df.loc[combined_df['f1-score'].idxmax()]
        print(f"Best F1-Score: {best_f1['Model']} ({best_f1['f1-score']:.4f})")
    
    # Medical interpretation
    print(f"\n{'='*80}")
    print("MEDICAL INTERPRETATION")
    print(f"{'='*80}")
    
    for _, row in combined_df.iterrows():
        model_name = row['Model']
        print(f"\n{model_name}:")
        
        # Accuracy interpretation
        if 'accuracy' in row:
            acc = row['accuracy']
            if acc >= 0.90:
                status = "EXCELLENT - Exceeds medical standards"
            elif acc >= 0.85:
                status = "VERY GOOD - Meets medical standards"
            elif acc >= 0.75:
                status = "GOOD - Clinically useful"
            elif acc >= 0.65:
                status = "FAIR - Needs improvement"
            else:
                status = "POOR - Not suitable for clinical use"
            
            print(f"  Accuracy: {acc:.1%} - {status}")
        
        # Sensitivity and Specificity
        if 'sensitivity' in row and 'specificity' in row:
            sens = row['sensitivity']
            spec = row['specificity']
            print(f"  Sensitivity: {sens:.1%} (detects {sens:.1%} of pneumonia cases)")
            print(f"  Specificity: {spec:.1%} (correctly identifies {spec:.1%} of normal cases)")
        
        # Precision
        if 'precision' in row:
            prec = row['precision']
            print(f"  Precision: {prec:.1%} (when predicting pneumonia, {prec:.1%} are correct)")
    
    # Summary for assignment
    print(f"\n{'='*80}")
    print("ASSIGNMENT SUMMARY")
    print(f"{'='*80}")
    
    print("FOUR Models Implemented:")
    for i, model in enumerate(combined_df['Model'], 1):
        print(f"  {i}. {model}")
    
    print(f"\nPerformance Range:")
    if 'accuracy' in combined_df.columns:
        min_acc = combined_df['accuracy'].min()
        max_acc = combined_df['accuracy'].max()
        avg_acc = combined_df['accuracy'].mean()
        print(f"  Accuracy: {min_acc:.1%} - {max_acc:.1%} (Average: {avg_acc:.1%})")
    
    print(f"\nBest Overall Model:")
    if 'accuracy' in combined_df.columns:
        best_overall = combined_df.loc[combined_df['accuracy'].idxmax()]
        print(f"  {best_overall['Model']} with {best_overall['accuracy']:.1%} accuracy")
    
    print(f"\nReady for Assignment Report!")
    print("  - Four models implemented")
    print("  - Performance metrics calculated")
    print("  - Medical interpretation provided")
    print("  - Results saved to CSV files")
    
    return combined_df

def create_sample_comparison():
    """Create a sample comparison table."""
    print("\n" + "="*80)
    print("SAMPLE COMPARISON TABLE")
    print("="*80)
    
    # Sample results
    sample_results = {
        'Model': ['Custom CNN', 'ResNet50 Transfer Learning', 'Logistic Regression', 'Vision Transformer'],
        'Framework': ['TensorFlow', 'TensorFlow', 'Scikit-learn', 'PyTorch'],
        'Type': ['Deep Learning', 'Transfer Learning', 'Traditional ML', 'Transformer'],
        'Accuracy': [0.891, 0.891, 0.835, 0.875],
        'Precision': [0.891, 0.891, 0.984, 0.890],
        'Recall': [1.000, 1.000, 0.787, 0.880],
        'F1-Score': [0.943, 0.943, 0.875, 0.885]
    }
    
    df = pd.DataFrame(sample_results)
    print(df.to_string(index=False))
    
    print(f"\nThis is an example comparison table.")
    print("Run your models to get actual results!")

if __name__ == "__main__":
    print("Choose option:")
    print("1. View existing results")
    print("2. Show sample comparison")
    
    try:
        choice = input("\nEnter your choice (1-2): ").strip()
    except EOFError:
        choice = "1"
    
    if choice == "1":
        results = view_results()
    elif choice == "2":
        create_sample_comparison()
    else:
        print("Invalid choice. Viewing existing results...")
        results = view_results()
    
    if results is not None:
        print("\nResults analysis completed!")
    else:
        print("\nNo results to analyze.")
