"""
Setup script for Pneumonia Detection System
Automatically installs dependencies and sets up the project
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("ERROR: Python 3.8+ is required. Current version:", f"{version.major}.{version.minor}")
        return False
    print(f"SUCCESS: Python {version.major}.{version.minor} detected")
    return True

def install_requirements():
    """Install required packages."""
    print("\nInstalling requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("SUCCESS: Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install requirements: {e}")
        return False

def check_gpu_status():
    """Check GPU availability."""
    print("\nChecking GPU status...")
    
    # Check PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"SUCCESS: PyTorch CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("WARNING: PyTorch CUDA not available")
    except ImportError:
        print("WARNING: PyTorch not installed")
    
    # Check TensorFlow GPU
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"SUCCESS: TensorFlow GPU available: {len(gpus)} GPU(s)")
        else:
            print("WARNING: TensorFlow GPU not available")
    except ImportError:
        print("WARNING: TensorFlow not installed")

def create_directories():
    """Create necessary directories."""
    print("\nCreating directories...")
    directories = [
        "01_Custom_CNN/results",
        "02_ResNet50_Transfer_Learning/results",
        "03_Logistic_Regression/results",
        "04_Vision_Transformer/results",
        "plots",
        "results"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"SUCCESS: Created {directory}")

def test_installation():
    """Test if the installation works."""
    print("\nTesting installation...")
    try:
        # Test imports
        import tensorflow as tf
        import torch
        import sklearn
        import streamlit
        import cv2
        import PIL
        import matplotlib
        import seaborn
        import pandas
        import numpy
        
        print("SUCCESS: All packages imported successfully")
        return True
    except ImportError as e:
        print(f"ERROR: Import error: {e}")
        return False

def show_next_steps():
    """Show next steps to the user."""
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Download the dataset:")
    print("   - Go to: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
    print("   - Extract to 'data/' folder")
    print("   - Ensure structure: data/train/NORMAL/, data/train/PNEUMONIA/, etc.")
    print("\n2. Run the project:")
    print("   python main.py")
    print("\n3. Quick results (no dataset needed):")
    print("   python quick_comparison.py")
    print("\n4. Web interface:")
    print("   streamlit run app.py")
    print("\n5. For GitHub upload:")
    print("   - Follow GITHUB_SETUP.md guide")
    print("   - Upload to GitHub repository")

def main():
    """Main setup function."""
    print("PNEUMONIA DETECTION SYSTEM - SETUP")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("\nERROR: Setup failed. Please install requirements manually:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Check GPU status
    check_gpu_status()
    
    # Test installation
    if not test_installation():
        print("\nERROR: Setup incomplete. Please check error messages above.")
        sys.exit(1)
    
    # Show next steps
    show_next_steps()

if __name__ == "__main__":
    main()
