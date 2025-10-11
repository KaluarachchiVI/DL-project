"""
Simple GPU Test Script
Test GPU acceleration for all frameworks
"""

import time
import os

def test_pytorch_gpu():
    """Test PyTorch GPU acceleration."""
    print("Testing PyTorch GPU...")
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"PyTorch CUDA available: True")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
            # Test tensor operations
            start_time = time.time()
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.mm(x, y)
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time
            
            print(f"GPU test completed in: {gpu_time:.4f} seconds")
            return True
        else:
            print("PyTorch CUDA: Not available")
            return False
    except Exception as e:
        print(f"PyTorch GPU test failed: {e}")
        return False

def test_tensorflow_gpu():
    """Test TensorFlow GPU acceleration."""
    print("\nTesting TensorFlow GPU...")
    try:
        import tensorflow as tf
        
        print(f"TensorFlow version: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        print(f"TensorFlow GPU available: {len(gpus)} GPU(s)")
        
        if gpus:
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu}")
            
            # Test tensor operations
            start_time = time.time()
            with tf.device('/GPU:0'):
                x = tf.random.normal([1000, 1000])
                y = tf.random.normal([1000, 1000])
                z = tf.matmul(x, y)
            tf_time = time.time() - start_time
            
            print(f"GPU test completed in: {tf_time:.4f} seconds")
            return True
        else:
            print("TensorFlow GPU: Not available")
            return False
    except Exception as e:
        print(f"TensorFlow GPU test failed: {e}")
        return False

def test_model_speed():
    """Test model training speed."""
    print("\nTesting model training speed...")
    
    # Test Custom CNN
    try:
        os.chdir("01_Custom_CNN")
        from gpu_cnn import GPUOptimizedCNN
        
        print("Testing Custom CNN GPU acceleration...")
        start_time = time.time()
        
        # Create model
        cnn = GPUOptimizedCNN()
        cnn.compile_model()
        
        model_time = time.time() - start_time
        print(f"Custom CNN model creation: {model_time:.4f} seconds")
        
        os.chdir("..")
        return True
    except Exception as e:
        print(f"Custom CNN test failed: {e}")
        os.chdir("..")
        return False

def main():
    """Main GPU test function."""
    print("="*80)
    print("GPU ACCELERATION TEST")
    print("="*80)
    
    # Test PyTorch
    pytorch_ok = test_pytorch_gpu()
    
    # Test TensorFlow
    tensorflow_ok = test_tensorflow_gpu()
    
    # Test model speed
    model_ok = test_model_speed()
    
    # Summary
    print("\n" + "="*80)
    print("GPU TEST SUMMARY")
    print("="*80)
    
    if pytorch_ok:
        print("PyTorch GPU: WORKING")
    else:
        print("PyTorch GPU: NOT WORKING")
    
    if tensorflow_ok:
        print("TensorFlow GPU: WORKING")
    else:
        print("TensorFlow GPU: NOT WORKING")
    
    if model_ok:
        print("Model GPU: WORKING")
    else:
        print("Model GPU: NOT WORKING")
    
    if pytorch_ok or tensorflow_ok:
        print("\nGPU acceleration is available!")
        print("You can use the GPU-optimized models for faster training.")
    else:
        print("\nGPU acceleration is not available.")
        print("Training will use CPU (slower but still functional).")

if __name__ == "__main__":
    main()
