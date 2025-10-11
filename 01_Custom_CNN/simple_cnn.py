"""
Simple Custom CNN for Pneumonia Detection
TensorFlow/Keras implementation
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

class SimpleCNN:
    """Simple CNN for pneumonia detection."""
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
    
    def build_model(self):
        """Build simple CNN architecture."""
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
        """Compile the model."""
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, train_generator, val_generator, epochs=20):
        """Train the model."""
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint('results/custom_cnn_best.h5', monitor='val_accuracy', save_best_only=True)
        ]
        
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self, test_generator):
        """Evaluate the model."""
        results = self.model.evaluate(test_generator, verbose=0)
        return {
            'loss': results[0],
            'accuracy': results[1]
        }

def train_custom_cnn():
    """Train Custom CNN model."""
    print("="*60)
    print("CUSTOM CNN TRAINING")
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
    cnn = SimpleCNN()
    cnn.build_model()
    cnn.compile_model()
    
    print("Model Architecture:")
    cnn.model.summary()
    
    # Train model
    print("\nTraining Custom CNN...")
    history = cnn.train(train_generator, val_generator, epochs=20)
    
    # Evaluate
    results = cnn.evaluate(val_generator)
    print(f"\nResults: Accuracy = {results['accuracy']:.4f}")
    
    return cnn, results

if __name__ == "__main__":
    train_custom_cnn()
