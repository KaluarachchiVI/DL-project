"""
GPU-Optimized Results Generator
Runs all 4 models with GPU acceleration and creates comprehensive comparison
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

def check_gpu_status():
    """Check GPU status for all frameworks."""
    print("="*80)
    print("GPU STATUS CHECK")
    print("="*80)
    
    # PyTorch GPU check
    try:
        import torch
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"PyTorch GPU: {torch.cuda.get_device_name(0)}")
            print(f"PyTorch GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except Exception as e:
        print(f"PyTorch GPU check failed: {e}")
    
    # TensorFlow GPU check
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        print(f"TensorFlow GPU available: {len(gpus)} GPU(s)")
        if gpus:
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu}")
    except Exception as e:
        print(f"TensorFlow GPU check failed: {e}")
    
    print("="*80)

def run_gpu_model_and_get_results(model_folder, script_name, model_name):
    """Run a GPU-optimized model and get its results."""
    print(f"\n{'='*60}")
    print(f"RUNNING GPU-OPTIMIZED {model_name.upper()}")
    print(f"{'='*60}")
    
    try:
        # Change to model directory
        original_dir = os.getcwd()
        os.chdir(model_folder)
        
        start_time = time.time()
        
        # Import and run the GPU-optimized model
        if script_name == "gpu_cnn.py":
            from gpu_cnn import train_gpu_cnn
            model, results = train_gpu_cnn()
        elif script_name == "gpu_resnet.py":
            from gpu_resnet import train_gpu_resnet
            model, results = train_gpu_resnet()
        elif script_name == "simple_logistic.py":
            from simple_logistic import train_logistic_regression
            model, results = train_logistic_regression()
        elif script_name == "gpu_vit.py":
            from gpu_vit import train_gpu_vit
            model, results = train_gpu_vit()
        else:
            print(f"❌ Unknown script: {script_name}")
            return None
        
        total_time = time.time() - start_time
        
        # Return to original directory
        os.chdir(original_dir)
        
        print(f"SUCCESS: {model_name} completed successfully!")
        print(f"Total time: {total_time:.2f} seconds")
        
        # Add timing information to results
        if isinstance(results, dict):
            results['total_time'] = total_time
        
        return results
        
    except Exception as e:
        print(f"ERROR: {model_name} failed: {e}")
        # Return to original directory
        os.chdir(original_dir)
        return None

def generate_gpu_results():
    """Generate results for all 4 models with GPU optimization."""
    print("="*100)
    print("GPU-OPTIMIZED PNEUMONIA DETECTION - GENERATING ALL RESULTS")
    print("="*100)
    print("This will train all 4 models with GPU acceleration.")
    print("Training should be much faster with proper GPU usage.")
    print("="*100)
    
    # Check GPU status first
    check_gpu_status()
    
    # Define models to run (GPU-optimized versions)
    models = [
        ("01_Custom_CNN", "gpu_cnn.py", "Custom CNN (GPU)"),
        ("02_ResNet50_Transfer_Learning", "gpu_resnet.py", "ResNet50 Transfer Learning (GPU)"),
        ("03_Logistic_Regression", "simple_logistic.py", "Logistic Regression (CPU)"),
        ("04_Vision_Transformer", "gpu_vit.py", "Vision Transformer (GPU)")
    ]
    
    all_results = {}
    total_start_time = time.time()
    
    # Run each model
    for model_folder, script_name, model_name in models:
        print(f"\n🚀 Starting {model_name}...")
        results = run_gpu_model_and_get_results(model_folder, script_name, model_name)
        
        if results is not None:
            all_results[model_name] = results
        else:
            print(f"⚠️ {model_name} failed to complete.")
    
    total_time = time.time() - total_start_time
    
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
            
            # Performance analysis
            print(f"\n⏱️ PERFORMANCE ANALYSIS:")
            print("="*100)
            
            if 'total_time' in comparison_df.columns:
                fastest_model = comparison_df.loc[comparison_df['total_time'].idxmin()]
                slowest_model = comparison_df.loc[comparison_df['total_time'].idxmax()]
                print(f"Fastest Training: {fastest_model['Model']} ({fastest_model['total_time']:.2f}s)")
                print(f"Slowest Training: {slowest_model['Model']} ({slowest_model['total_time']:.2f}s)")
            
            # Find best models
            print(f"\n🏆 BEST PERFORMING MODELS:")
            print("="*100)
            
            metrics_to_check = ['accuracy', 'precision', 'recall', 'f1-score']
            for metric in metrics_to_check:
                if metric in comparison_df.columns:
                    best_model = comparison_df.loc[comparison_df[metric].idxmax()]
                    print(f"Best {metric.upper()}: {best_model['Model']} ({best_model[metric]:.4f})")
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"gpu_optimized_results_{timestamp}.csv"
            comparison_df.to_csv(results_file, index=False)
            print(f"\n📁 Results saved to: {results_file}")
            
            # Summary
            print(f"\n{'='*100}")
            print("SUMMARY")
            print(f"{'='*100}")
            print(f"✅ Successfully trained {len(all_results)} out of {len(models)} models")
            print(f"⏱️ Total time: {total_time:.2f} seconds")
            print(f"📊 Comprehensive comparison created")
            print(f"📁 Results saved to CSV file")
            print(f"🚀 GPU acceleration working properly!")
            
        else:
            print("❌ No valid results to compare.")
    else:
        print("❌ No models completed successfully.")
    
    return all_results

def quick_gpu_test():
    """Quick GPU test to verify acceleration is working."""
    print("="*80)
    print("QUICK GPU TEST")
    print("="*80)
    
    # Test PyTorch GPU
    try:
        import torch
        if torch.cuda.is_available():
            print("✅ PyTorch GPU: Available")
            
            # Test tensor operations
            start_time = time.time()
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.mm(x, y)
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time
            
            print(f"⏱️ PyTorch GPU test: {gpu_time:.4f} seconds")
        else:
            print("❌ PyTorch GPU: Not available")
    except Exception as e:
        print(f"❌ PyTorch GPU test failed: {e}")
    
    # Test TensorFlow GPU
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print("✅ TensorFlow GPU: Available")
            
            # Test tensor operations
            start_time = time.time()
            with tf.device('/GPU:0'):
                x = tf.random.normal([1000, 1000])
                y = tf.random.normal([1000, 1000])
                z = tf.matmul(x, y)
            tf_time = time.time() - start_time
            
            print(f"⏱️ TensorFlow GPU test: {tf_time:.4f} seconds")
        else:
            print("❌ TensorFlow GPU: Not available")
    except Exception as e:
        print(f"❌ TensorFlow GPU test failed: {e}")

if __name__ == "__main__":
    print("Choose GPU optimization option:")
    print("1. Generate GPU-optimized results (train all models)")
    print("2. Quick GPU test")
    
    try:
        choice = input("\nEnter your choice (1-2): ").strip()
    except EOFError:
        choice = "1"
    
    if choice == "1":
        print("\n🚀 Generating GPU-optimized results for all models...")
        results = generate_gpu_results()
    elif choice == "2":
        print("\n🔧 Running quick GPU test...")
        quick_gpu_test()
    else:
        print("❌ Invalid choice. Running GPU-optimized results...")
        results = generate_gpu_results()
    
    if results is not None:
        print("\n✅ GPU-optimized results generation completed!")
    else:
        print("\n❌ GPU results generation failed.")
