"""
Fix TensorFlow GPU Configuration
Ensure TensorFlow can use GPU properly
"""

import os
import subprocess
import sys

def install_tensorflow_gpu():
    """Install TensorFlow with GPU support."""
    print("Installing TensorFlow with GPU support...")
    
    try:
        # Uninstall current TensorFlow
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "tensorflow", "-y"], check=True)
        
        # Install TensorFlow with GPU support
        subprocess.run([sys.executable, "-m", "pip", "install", "tensorflow[and-cuda]"], check=True)
        
        print("TensorFlow with GPU support installed successfully!")
        return True
    except Exception as e:
        print(f"Failed to install TensorFlow GPU: {e}")
        return False

def configure_tensorflow_gpu():
    """Configure TensorFlow to use GPU."""
    print("Configuring TensorFlow GPU...")
    
    try:
        import tensorflow as tf
        
        # Check current GPU status
        gpus = tf.config.list_physical_devices('GPU')
        print(f"Current GPU devices: {gpus}")
        
        if gpus:
            # Configure memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth configured")
            
            # Test GPU
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
                print(f"GPU test result: {c}")
            
            return True
        else:
            print("No GPU devices found")
            return False
            
    except Exception as e:
        print(f"TensorFlow GPU configuration failed: {e}")
        return False

def main():
    """Main function to fix TensorFlow GPU."""
    print("="*80)
    print("TENSORFLOW GPU CONFIGURATION FIX")
    print("="*80)
    
    print("Current TensorFlow version:")
    try:
        import tensorflow as tf
        print(f"TensorFlow: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        print(f"GPU devices: {len(gpus)}")
    except Exception as e:
        print(f"TensorFlow import failed: {e}")
    
    print("\nOptions:")
    print("1. Install TensorFlow with GPU support")
    print("2. Configure existing TensorFlow for GPU")
    print("3. Skip (use CPU-only)")
    
    try:
        choice = input("\nEnter your choice (1-3): ").strip()
    except EOFError:
        choice = "2"
    
    if choice == "1":
        install_tensorflow_gpu()
    elif choice == "2":
        configure_tensorflow_gpu()
    elif choice == "3":
        print("Skipping GPU configuration. Using CPU-only.")
    else:
        print("Invalid choice. Configuring existing TensorFlow...")
        configure_tensorflow_gpu()

if __name__ == "__main__":
    main()
