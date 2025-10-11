"""
Simple Model Comparison Script
Compares all 4 models for pneumonia detection
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

def run_all_models():
    """Run all 4 models and compare results."""
    
    print("="*80)
    print("🫁 PNEUMONIA DETECTION - MODEL COMPARISON")
    print("="*80)
    print("Training and comparing all 4 models:")
    print("1. Custom CNN (TensorFlow/Keras)")
    print("2. ResNet50 Transfer Learning (TensorFlow/Keras)")
    print("3. Logistic Regression (Scikit-learn)")
    print("4. Vision Transformer (PyTorch)")
    print("="*80)
    
    results = {}
    
    # Model 1: Custom CNN
    print("\n" + "="*50)
    print("MODEL 1: CUSTOM CNN")
    print("="*50)
    try:
        os.chdir("01_Custom_CNN")
        from simple_cnn import train_custom_cnn
        model1, results1 = train_custom_cnn()
        results['Custom CNN'] = results1
        print("✅ Custom CNN completed successfully!")
        os.chdir("..")
    except Exception as e:
        print(f"❌ Custom CNN failed: {e}")
        os.chdir("..")
    
    # Model 2: ResNet50 Transfer Learning
    print("\n" + "="*50)
    print("MODEL 2: RESNET50 TRANSFER LEARNING")
    print("="*50)
    try:
        os.chdir("02_ResNet50_Transfer_Learning")
        from simple_resnet import train_resnet
        model2, results2 = train_resnet()
        results['ResNet50 Transfer Learning'] = results2
        print("✅ ResNet50 Transfer Learning completed successfully!")
        os.chdir("..")
    except Exception as e:
        print(f"❌ ResNet50 Transfer Learning failed: {e}")
        os.chdir("..")
    
    # Model 3: Logistic Regression
    print("\n" + "="*50)
    print("MODEL 3: LOGISTIC REGRESSION")
    print("="*50)
    try:
        os.chdir("03_Logistic_Regression")
        from simple_logistic import train_logistic_regression
        model3, results3 = train_logistic_regression()
        results['Logistic Regression'] = results3
        print("✅ Logistic Regression completed successfully!")
        os.chdir("..")
    except Exception as e:
        print(f"❌ Logistic Regression failed: {e}")
        os.chdir("..")
    
    # Model 4: Vision Transformer
    print("\n" + "="*50)
    print("MODEL 4: VISION TRANSFORMER")
    print("="*50)
    try:
        os.chdir("04_Vision_Transformer")
        from simple_vit import train_vision_transformer
        model4, results4 = train_vision_transformer()
        results['Vision Transformer'] = results4
        print("✅ Vision Transformer completed successfully!")
        os.chdir("..")
    except Exception as e:
        print(f"❌ Vision Transformer failed: {e}")
        os.chdir("..")
    
    # Compare results
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    
    if results:
        # Create comparison DataFrame
        comparison_data = []
        for model_name, model_results in results.items():
            if isinstance(model_results, dict):
                row = {'Model': model_name}
                for metric, value in model_results.items():
                    if isinstance(value, float):
                        row[metric] = value
                comparison_data.append(row)
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            # Display results
            print("\nModel Performance Comparison:")
            print(comparison_df.round(4).to_string(index=False))
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"model_comparison_results_{timestamp}.csv"
            comparison_df.to_csv(results_file, index=False)
            print(f"\nResults saved to: {results_file}")
            
            # Find best model
            if 'accuracy' in comparison_df.columns:
                best_model = comparison_df.loc[comparison_df['accuracy'].idxmax()]
                print(f"\n🏆 Best Model: {best_model['Model']} with {best_model['accuracy']:.4f} accuracy")
        else:
            print("No valid results to compare.")
    else:
        print("No models completed successfully.")
    
    print("\n" + "="*80)
    print("MODEL COMPARISON COMPLETED!")
    print("="*80)
    
    return results

if __name__ == "__main__":
    try:
        results = run_all_models()
        print("\nAll model training and comparison completed!")
    except Exception as e:
        print(f"Error during model comparison: {e}")
        import traceback
        traceback.print_exc()
