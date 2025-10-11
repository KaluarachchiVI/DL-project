"""
Generate Results for All Models
Runs all 4 models and creates comprehensive comparison
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def run_model_and_get_results(model_folder, script_name, model_name):
    """Run a model and get its results."""
    print(f"\n{'='*60}")
    print(f"RUNNING {model_name.upper()}")
    print(f"{'='*60}")
    
    try:
        # Change to model directory
        original_dir = os.getcwd()
        os.chdir(model_folder)
        
        # Import and run the model
        if script_name == "simple_cnn.py":
            from simple_cnn import train_custom_cnn
            model, results = train_custom_cnn()
        elif script_name == "simple_resnet.py":
            from simple_resnet import train_resnet
            model, results = train_resnet()
        elif script_name == "simple_logistic.py":
            from simple_logistic import train_logistic_regression
            model, results = train_logistic_regression()
        elif script_name == "simple_vit.py":
            from simple_vit import train_vision_transformer
            model, results = train_vision_transformer()
        else:
            print(f"❌ Unknown script: {script_name}")
            return None
        
        # Return to original directory
        os.chdir(original_dir)
        
        print(f"✅ {model_name} completed successfully!")
        return results
        
    except Exception as e:
        print(f"❌ {model_name} failed: {e}")
        # Return to original directory
        os.chdir(original_dir)
        return None

def generate_all_results():
    """Generate results for all 4 models."""
    print("="*100)
    print("🫁 PNEUMONIA DETECTION - GENERATING ALL RESULTS")
    print("="*100)
    print("This will train all 4 models and create a comprehensive comparison.")
    print("This may take some time. Please be patient...")
    print("="*100)
    
    # Define models to run
    models = [
        ("01_Custom_CNN", "simple_cnn.py", "Custom CNN"),
        ("02_ResNet50_Transfer_Learning", "simple_resnet.py", "ResNet50 Transfer Learning"),
        ("03_Logistic_Regression", "simple_logistic.py", "Logistic Regression"),
        ("04_Vision_Transformer", "simple_vit.py", "Vision Transformer")
    ]
    
    all_results = {}
    
    # Run each model
    for model_folder, script_name, model_name in models:
        print(f"\n🚀 Starting {model_name}...")
        results = run_model_and_get_results(model_folder, script_name, model_name)
        
        if results is not None:
            all_results[model_name] = results
        else:
            print(f"⚠️ {model_name} failed to complete.")
    
    # Create comprehensive comparison
    if all_results:
        print(f"\n{'='*100}")
        print("CREATING COMPREHENSIVE COMPARISON")
        print(f"{'='*100}")
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, results in all_results.items():
            if isinstance(results, dict):
                row = {'Model': model_name}
                for metric, value in results.items():
                    if isinstance(value, (int, float)):
                        row[metric] = value
                comparison_data.append(row)
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            # Display results
            print("\n📊 MODEL PERFORMANCE COMPARISON:")
            print("="*100)
            display_df = comparison_df.round(4)
            print(display_df.to_string(index=False))
            
            # Find best models
            print("\n🏆 BEST PERFORMING MODELS:")
            print("="*100)
            
            metrics_to_check = ['accuracy', 'precision', 'recall', 'f1-score']
            for metric in metrics_to_check:
                if metric in comparison_df.columns:
                    best_model = comparison_df.loc[comparison_df[metric].idxmax()]
                    print(f"Best {metric.upper()}: {best_model['Model']} ({best_model[metric]:.4f})")
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"comprehensive_model_results_{timestamp}.csv"
            comparison_df.to_csv(results_file, index=False)
            print(f"\n📁 Results saved to: {results_file}")
            
            # Medical interpretation
            print(f"\n{'='*100}")
            print("MEDICAL INTERPRETATION")
            print(f"{'='*100}")
            
            for _, row in comparison_df.iterrows():
                model_name = row['Model']
                if 'accuracy' in row:
                    acc = row['accuracy']
                    if acc >= 0.90:
                        status = "✅ EXCELLENT - Exceeds medical standards"
                    elif acc >= 0.85:
                        status = "✅ VERY GOOD - Meets medical standards"
                    elif acc >= 0.75:
                        status = "✅ GOOD - Clinically useful"
                    elif acc >= 0.65:
                        status = "⚠️ FAIR - Needs improvement"
                    else:
                        status = "❌ POOR - Not suitable for clinical use"
                    
                    print(f"{model_name}: {acc:.1%} accuracy - {status}")
            
            print(f"\n{'='*100}")
            print("SUMMARY")
            print(f"{'='*100}")
            print(f"✅ Successfully trained {len(all_results)} out of {len(models)} models")
            print(f"📊 Comprehensive comparison created")
            print(f"📁 Results saved to CSV file")
            print(f"🎯 Ready for assignment report!")
            
        else:
            print("❌ No valid results to compare.")
    else:
        print("❌ No models completed successfully.")
    
    return all_results

def quick_analysis():
    """Quick analysis of existing results."""
    print("="*100)
    print("🫁 PNEUMONIA DETECTION - QUICK ANALYSIS")
    print("="*100)
    
    # Check for existing results
    csv_files = [
        "model_comparison_results.csv",
        "accuracy_test_results.csv"
    ]
    
    all_data = []
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                print(f"📁 Found results in {csv_file}")
                all_data.append(df)
            except Exception as e:
                print(f"❌ Error reading {csv_file}: {e}")
    
    if all_data:
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        print("\n📊 EXISTING RESULTS SUMMARY:")
        print("="*100)
        print(combined_df.to_string(index=False))
        
        # Analysis
        if 'accuracy' in combined_df.columns:
            best_accuracy = combined_df.loc[combined_df['accuracy'].idxmax()]
            print(f"\n🏆 Best Accuracy: {best_accuracy['Model']} ({best_accuracy['accuracy']:.4f})")
        
        return combined_df
    else:
        print("❌ No existing results found.")
        return None

if __name__ == "__main__":
    print("Choose analysis option:")
    print("1. Generate new results (train all models)")
    print("2. Quick analysis (use existing results)")
    
    try:
        choice = input("\nEnter your choice (1-2): ").strip()
    except EOFError:
        choice = "2"
    
    if choice == "1":
        print("\n🚀 Generating new results for all models...")
        results = generate_all_results()
    elif choice == "2":
        print("\n📊 Analyzing existing results...")
        results = quick_analysis()
    else:
        print("❌ Invalid choice. Running quick analysis...")
        results = quick_analysis()
    
    if results is not None:
        print("\n✅ Analysis completed successfully!")
    else:
        print("\n❌ Analysis failed.")
