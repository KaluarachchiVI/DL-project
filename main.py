"""
Pneumonia Detection System - Main Interface
Four Models for Assignment: Custom CNN, ResNet50, Logistic Regression, Vision Transformer
"""

import os
import sys

def main():
    """Main interface."""
    
    print("="*80)
    print("PNEUMONIA DETECTION SYSTEM")
    print("="*80)
    print("Four Models for Assignment:")
    print("1. Custom CNN (TensorFlow/Keras)")
    print("2. ResNet50 Transfer Learning (TensorFlow/Keras)")
    print("3. Logistic Regression (Scikit-learn)")
    print("4. Vision Transformer (PyTorch)")
    print("="*80)
    print("Choose your option:")
    print("1. Train Individual Model")
    print("2. Launch Web Interface")
    print("3. View Results")
    print("4. Generate All Results")
    print("5. Compare Models")
    print("6. GPU-Optimized Results")
    print("7. Quick Comparison (Instant Results)")
    print("8. Exit")
    
    try:
        choice = input("\nEnter your choice (1-8): ").strip()
    except EOFError:
        print("\nRunning in non-interactive mode. Exiting...")
        return
    
    if choice == "1":
        train_individual_model()
    elif choice == "2":
        launch_web_interface()
    elif choice == "3":
        view_results()
    elif choice == "4":
        generate_all_results()
    elif choice == "5":
        compare_models()
    elif choice == "6":
        gpu_optimized_results()
    elif choice == "7":
        quick_comparison()
    elif choice == "8":
        print("Goodbye!")
        return
    else:
        print("Invalid choice. Please run the script again.")

def train_individual_model():
    """Train individual model."""
    print("\nChoose model to train:")
    print("1. Custom CNN")
    print("2. ResNet50 Transfer Learning")
    print("3. Logistic Regression")
    print("4. Vision Transformer")
    
    try:
        model_choice = input("\nEnter model number (1-4): ").strip()
    except EOFError:
        return
    
    model_folders = {
        "1": ("01_Custom_CNN", "Custom CNN"),
        "2": ("02_ResNet50_Transfer_Learning", "ResNet50 Transfer Learning"),
        "3": ("03_Logistic_Regression", "Logistic Regression"),
        "4": ("04_Vision_Transformer", "Vision Transformer")
    }
    
    if model_choice in model_folders:
        folder, name = model_folders[model_choice]
        print(f"\n🚀 Training {name}...")
        print(f"Navigate to {folder}/ and run the training script")
        print(f"Example: cd {folder} && python train_*.py")
    else:
        print("❌ Invalid choice")

def launch_web_interface():
    """Launch web interface."""
    print("\n🌐 Launching Web Interface...")
    print("Opening Streamlit app in your browser...")
    os.system("streamlit run app.py")

def view_results():
    """View existing results."""
    print("\n📊 Viewing Results...")
    os.system("python view_results.py")

def generate_all_results():
    """Generate results for all models."""
    print("\n🚀 Generating All Results...")
    os.system("python generate_results.py")

def compare_models():
    """Compare models and create analysis."""
    print("\n📊 Comparing Models...")
    os.system("python model_comparison_analysis.py")

def gpu_optimized_results():
    """Generate GPU-optimized results."""
    print("\n🚀 Generating GPU-Optimized Results...")
    os.system("python gpu_results_generator.py")

def quick_comparison():
    """Generate quick comparison results."""
    print("\n⚡ Generating Quick Comparison...")
    os.system("python quick_comparison.py")

if __name__ == "__main__":
    main()