"""
GPU-Optimized Custom CNN for Pneumonia Detection
TensorFlow/Keras with proper GPU configuration
"""

import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import time

# Configure GPU for TensorFlow
def setup_tensorflow_gpu():
    """Configure TensorFlow to use GPU properly."""
    # Set memory growth to avoid allocating all GPU memory at once
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ TensorFlow GPU configured: {len(gpus)} GPU(s) available")
            return True
        except RuntimeError as e:
            print(f"❌ GPU setup error: {e}")
            return False
    else:
        print("⚠️ No GPU detected for TensorFlow")
        return False

# Initialize GPU configuration
setup_tensorflow_gpu()

class GPUOptimizedCNN:
    """GPU-optimized CNN for pneumonia detection."""
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
        # Ensure GPU usage
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            self.build_model()
    
    def build_model(self):
        """Build optimized CNN architecture."""
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            self.model = Sequential([
                # First Conv Block
                Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Dropout(0.25),
                
                # Second Conv Block
                Conv2D(64, (3, 3), activation='relu'),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Dropout(0.25),
                
                # Third Conv Block
                Conv2D(128, (3, 3), activation='relu'),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Dropout(0.25),
                
                # Flatten and Dense
                Flatten(),
                Dense(512, activation='relu'),
                Dropout(0.5),
                Dense(self.num_classes, activation='softmax')
            ])
        
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with GPU optimization."""
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            self.model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
    
    def train(self, train_generator, val_generator, epochs=10):
        """Train the model with GPU acceleration."""
        print("🚀 Training with GPU acceleration...")
        start_time = time.time()
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ModelCheckpoint('results/gpu_cnn_best.h5', monitor='val_accuracy', save_best_only=True)
        ]
        
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            history = self.model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
        
        training_time = time.time() - start_time
        print(f"⏱️ Training completed in {training_time:.2f} seconds")
        
        return history
    
    def evaluate(self, test_generator):
        """Evaluate the model."""
        start_time = time.time()
        
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            results = self.model.evaluate(test_generator, verbose=0)
        
        eval_time = time.time() - start_time
        print(f"⏱️ Evaluation completed in {eval_time:.2f} seconds")
        
        return {
            'loss': results[0],
            'accuracy': results[1],
            'eval_time': eval_time
        }

def train_gpu_cnn():
    """Train GPU-optimized Custom CNN model."""
    print("="*60)
    print("GPU-OPTIMIZED CUSTOM CNN TRAINING")
    print("="*60)
    
    # Import data preprocessing
    import sys
    sys.path.append('..')
    from utils.data_preprocessing import DataPreprocessor
    
    # Load data
    preprocessor = DataPreprocessor("../data", (224, 224), 32, 42)
    image_paths, labels = preprocessor.load_data_paths()
    X_train, X_val, y_train, y_val = preprocessor.create_train_val_split(image_paths, labels)
    
    # Create data generators
    train_generator, val_generator = preprocessor.get_tensorflow_generators(
        X_train, X_val, y_train, y_val, augmentation=True
    )
    
    # Create and train model
    cnn = GPUOptimizedCNN()
    cnn.compile_model()
    
    print("Model Architecture:")
    cnn.model.summary()
    
    # Train model
    print("\n🚀 Training GPU-Optimized Custom CNN...")
    history = cnn.train(train_generator, val_generator, epochs=10)
    
    # Evaluate
    results = cnn.evaluate(val_generator)
    print(f"\nResults: Accuracy = {results['accuracy']:.4f}")
    print(f"Evaluation time: {results['eval_time']:.2f} seconds")
    
    return cnn, results

if __name__ == "__main__":
    train_gpu_cnn()
