"""
Custom CNN model for chest X-ray pneumonia classification using TensorFlow/Keras.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
    BatchNormalization, GlobalAveragePooling2D, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import os
from utils.evaluation import ModelEvaluator
from utils.visualization import Visualizer

class CustomCNN:
    """Custom CNN model for chest X-ray classification."""
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=2, random_seed=42):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.random_seed = random_seed
        self.model = None
        self.history = None
        self.evaluator = ModelEvaluator()
        self.visualizer = Visualizer()
        
        # Set random seed
        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)
    
    def build_model(self, architecture='standard'):
        """Build the CNN model architecture."""
        
        if architecture == 'standard':
            return self._build_standard_cnn()
        elif architecture == 'deep':
            return self._build_deep_cnn()
        elif architecture == 'lightweight':
            return self._build_lightweight_cnn()
        else:
            raise ValueError("Architecture must be 'standard', 'deep', or 'lightweight'")
    
    def _build_standard_cnn(self):
        """Build standard CNN architecture."""
        model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Fourth Convolutional Block
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Global Average Pooling
            GlobalAveragePooling2D(),
            
            # Dense layers
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            
            # Output layer
            Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def _build_deep_cnn(self):
        """Build deeper CNN architecture."""
        model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Fourth Convolutional Block
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Fifth Convolutional Block
            Conv2D(512, (3, 3), activation='relu'),
            BatchNormalization(),
            Conv2D(512, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Global Average Pooling
            GlobalAveragePooling2D(),
            
            # Dense layers
            Dense(1024, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            
            # Output layer
            Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def _build_lightweight_cnn(self):
        """Build lightweight CNN architecture."""
        model = Sequential([
            # First Convolutional Block
            Conv2D(16, (3, 3), activation='relu', input_shape=self.input_shape),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Second Convolutional Block
            Conv2D(32, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Third Convolutional Block
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Global Average Pooling
            GlobalAveragePooling2D(),
            
            # Dense layers
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            
            # Output layer
            Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def compile_model(self, learning_rate=0.001, optimizer='adam'):
        """Compile the model."""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        if optimizer == 'adam':
            opt = Adam(learning_rate=learning_rate)
        else:
            opt = optimizer
        
        self.model.compile(
            optimizer=opt,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model compiled successfully!")
    
    def train(self, train_generator, val_generator, epochs=50, batch_size=32,
              callbacks=None, verbose=1):
        """Train the model."""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        # Default callbacks
        if callbacks is None:
            callbacks = self._get_default_callbacks()
        
        print("Training Custom CNN model...")
        
        # Train the model
        self.history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        print("Custom CNN training completed!")
        return self.history
    
    def _get_default_callbacks(self):
        """Get default training callbacks."""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'models/saved/custom_cnn_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        return callbacks
    
    def evaluate(self, test_generator):
        """Evaluate the model."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Get predictions
        y_pred_proba = self.model.predict(test_generator)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Get true labels
        y_true = test_generator.classes
        
        # Evaluate
        results = self.evaluator.evaluate_binary_classification(
            y_true, y_pred, y_pred_proba[:, 1], "Custom CNN"
        )
        
        return results
    
    def predict(self, test_generator):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        predictions = self.model.predict(test_generator)
        return predictions
    
    def plot_training_history(self):
        """Plot training history."""
        if self.history is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        self.visualizer.plot_training_history(
            self.history, "Custom CNN"
        )
    
    def save_model(self, filepath="models/saved/custom_cnn_model.h5"):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath="models/saved/custom_cnn_model.h5"):
        """Load a trained model."""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def get_model_summary(self):
        """Get model summary."""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        return self.model.summary()

def train_custom_cnn(train_generator, val_generator, test_generator,
                    architecture='standard', epochs=50):
    """Train and evaluate custom CNN model."""
    
    # Initialize CNN
    cnn = CustomCNN()
    
    # Build model
    cnn.build_model(architecture=architecture)
    cnn.compile_model()
    
    # Print model summary
    print("Model Architecture:")
    cnn.get_model_summary()
    
    # Train model
    history = cnn.train(train_generator, val_generator, epochs=epochs)
    
    # Plot training history
    cnn.plot_training_history()
    
    # Evaluate model
    results = cnn.evaluate(test_generator)
    
    # Save model
    cnn.save_model()
    
    return cnn, results, history

if __name__ == "__main__":
    # Test the custom CNN
    print("Custom CNN model implementation loaded successfully!")
