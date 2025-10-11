"""
GPU-Optimized ResNet50 Transfer Learning for Pneumonia Detection
TensorFlow/Keras with proper GPU configuration
"""

import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import time

# Configure GPU for TensorFlow
def setup_tensorflow_gpu():
    """Configure TensorFlow to use GPU properly."""
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

class GPUOptimizedResNet:
    """GPU-optimized ResNet50 transfer learning for pneumonia detection."""
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
        # Ensure GPU usage
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            self.build_model()
    
    def build_model(self):
        """Build ResNet50 transfer learning model with GPU optimization."""
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            # Load pre-trained ResNet50
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
            
            # Freeze base model
            base_model.trainable = False
            
            # Add custom classification head
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.3)(x)
            predictions = Dense(self.num_classes, activation='softmax')(x)
            
            # Create model
            self.model = Model(inputs=base_model.input, outputs=predictions)
        
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
            ModelCheckpoint('results/gpu_resnet_best.h5', monitor='val_accuracy', save_best_only=True)
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

def train_gpu_resnet():
    """Train GPU-optimized ResNet50 model."""
    print("="*60)
    print("GPU-OPTIMIZED RESNET50 TRANSFER LEARNING")
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
    resnet = GPUOptimizedResNet()
    resnet.compile_model()
    
    print("Model Architecture:")
    resnet.model.summary()
    
    # Train model
    print("\n🚀 Training GPU-Optimized ResNet50...")
    history = resnet.train(train_generator, val_generator, epochs=10)
    
    # Evaluate
    results = resnet.evaluate(val_generator)
    print(f"\nResults: Accuracy = {results['accuracy']:.4f}")
    print(f"Evaluation time: {results['eval_time']:.2f} seconds")
    
    return resnet, results

if __name__ == "__main__":
    train_gpu_resnet()
