"""
ESSENTIAL WORKFLOW FOR PNEUMONIA DETECTION PROJECT
==================================================

This script provides the essential workflow for:
1. Training all 4 models
2. Optimizing performance
3. Generating results
4. Comparing models

USAGE:
    python ESSENTIAL_WORKFLOW_CLEAN.py

OPTIONS:
    1. Train All Models (Recommended)
    2. Train Individual Model
    3. Generate Results Only
    4. Compare Models
    5. Launch Web Interface
    6. Exit
"""

import os
import sys
import time
import subprocess
from datetime import datetime

def print_header():
    """Print project header."""
    print("="*80)
    print("PNEUMONIA DETECTION - ESSENTIAL WORKFLOW")
    print("="*80)
    print("Four Models for Assignment:")
    print("1. Custom CNN (TensorFlow/Keras)")
    print("2. ResNet50 Transfer Learning (TensorFlow/Keras)")
    print("3. Logistic Regression (Scikit-learn)")
    print("4. Vision Transformer (PyTorch)")
    print("="*80)

def check_requirements():
    """Check if all requirements are met."""
    print("\nChecking Requirements...")
    
    # Check if data exists
    if not os.path.exists("data"):
        print("Data directory not found!")
        return False
    
    # Check if balanced_data exists
    if not os.path.exists("balanced_data"):
        print("Balanced data directory not found!")
        return False
    
    # Check if model directories exist
    model_dirs = ["01_Custom_CNN", "02_ResNet50_Transfer_Learning", 
                  "03_Logistic_Regression", "04_Vision_Transformer"]
    
    for dir_name in model_dirs:
        if not os.path.exists(dir_name):
            print(f"{dir_name} directory not found!")
            return False
    
    print("All requirements met!")
    return True

def train_all_models():
    """Train all 4 models sequentially."""
    print("\nTRAINING ALL MODELS")
    print("="*50)
    
    models = [
        ("01_Custom_CNN", "Custom CNN", "simple_cnn.py"),
        ("02_ResNet50_Transfer_Learning", "ResNet50 Transfer Learning", "fixed_resnet.py"),
        ("03_Logistic_Regression", "Logistic Regression", "simple_logistic.py"),
        ("04_Vision_Transformer", "Vision Transformer", "simple_vit.py")
    ]
    
    results = {}
    
    for i, (model_dir, model_name, script_name) in enumerate(models, 1):
        print(f"\nTraining {i}/4: {model_name}")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            # Change to model directory and run training
            os.chdir(model_dir)
            result = subprocess.run([sys.executable, script_name], 
                                  capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"SUCCESS: {model_name} trained successfully!")
                results[model_name] = "Success"
            else:
                print(f"FAILED: {model_name} failed: {result.stderr}")
                results[model_name] = "Failed"
            
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT: {model_name} timed out (5 minutes)")
            results[model_name] = "Timeout"
        except Exception as e:
            print(f"ERROR: {model_name} error: {str(e)}")
            results[model_name] = "Error"
        finally:
            # Return to main directory
            os.chdir("..")
        
        end_time = time.time()
        print(f"Training time: {end_time - start_time:.1f} seconds")
    
    # Print summary
    print("\nTRAINING SUMMARY")
    print("="*50)
    for model_name, status in results.items():
        status_icon = "SUCCESS" if status == "Success" else "FAILED"
        print(f"{status_icon}: {model_name}: {status}")
    
    return results

def train_individual_model():
    """Train individual model."""
    print("\nTRAIN INDIVIDUAL MODEL")
    print("="*50)
    
    models = [
        ("01_Custom_CNN", "Custom CNN", "simple_cnn.py"),
        ("02_ResNet50_Transfer_Learning", "ResNet50 Transfer Learning", "fixed_resnet.py"),
        ("03_Logistic_Regression", "Logistic Regression", "simple_logistic.py"),
        ("04_Vision_Transformer", "Vision Transformer", "simple_vit.py")
    ]
    
    print("Available models:")
    for i, (_, model_name, _) in enumerate(models, 1):
        print(f"{i}. {model_name}")
    
    try:
        choice = int(input("\nEnter model number (1-4): ")) - 1
        if 0 <= choice < len(models):
            model_dir, model_name, script_name = models[choice]
            
            print(f"\nTraining {model_name}...")
            print("-" * 40)
            
            start_time = time.time()
            
            # Change to model directory and run training
            os.chdir(model_dir)
            result = subprocess.run([sys.executable, script_name], 
                                  capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"SUCCESS: {model_name} trained successfully!")
            else:
                print(f"FAILED: {model_name} failed: {result.stderr}")
            
            # Return to main directory
            os.chdir("..")
            
            end_time = time.time()
            print(f"Training time: {end_time - start_time:.1f} seconds")
        else:
            print("Invalid choice!")
    except ValueError:
        print("Invalid input!")
    except KeyboardInterrupt:
        print("\nTraining cancelled by user")

def generate_results():
    """Generate results for all models."""
    print("\nGENERATING RESULTS")
    print("="*50)
    
    try:
        # Run quick comparison to generate results
        result = subprocess.run([sys.executable, "quick_comparison.py"], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("SUCCESS: Results generated successfully!")
            print("Check the generated CSV files for results")
        else:
            print(f"FAILED: Failed to generate results: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("TIMEOUT: Results generation timed out")
    except Exception as e:
        print(f"ERROR: Error generating results: {str(e)}")

def compare_models():
    """Compare models and create analysis."""
    print("\nCOMPARING MODELS")
    print("="*50)
    
    try:
        # Run model comparison analysis
        result = subprocess.run([sys.executable, "model_comparison_analysis.py"], 
                              capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("SUCCESS: Model comparison completed!")
            print("Check the generated plots and analysis files")
        else:
            print(f"FAILED: Failed to compare models: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("TIMEOUT: Model comparison timed out")
    except Exception as e:
        print(f"ERROR: Error comparing models: {str(e)}")

def launch_web_interface():
    """Launch Streamlit web interface."""
    print("\nLAUNCHING WEB INTERFACE")
    print("="*50)
    
    try:
        print("Starting Streamlit app...")
        print("The web interface will open in your browser")
        print("Press Ctrl+C to stop the server")
        
        # Launch Streamlit
        subprocess.run(["streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\nWeb interface stopped by user")
    except Exception as e:
        print(f"ERROR: Error launching web interface: {str(e)}")

def main():
    """Main workflow interface."""
    print_header()
    
    # Check requirements
    if not check_requirements():
        print("\nRequirements not met. Please check your setup.")
        return
    
    while True:
        print("\nESSENTIAL WORKFLOW OPTIONS:")
        print("1. Train All Models (Recommended)")
        print("2. Train Individual Model")
        print("3. Generate Results Only")
        print("4. Compare Models")
        print("5. Launch Web Interface")
        print("6. Exit")
        
        try:
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == "1":
                train_all_models()
            elif choice == "2":
                train_individual_model()
            elif choice == "3":
                generate_results()
            elif choice == "4":
                compare_models()
            elif choice == "5":
                launch_web_interface()
            elif choice == "6":
                print("\nGoodbye!")
                break
            else:
                print("Invalid choice. Please enter 1-6.")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"ERROR: {str(e)}")

if __name__ == "__main__":
    main()
