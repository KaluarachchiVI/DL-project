"""
Simple ResNet50 Transfer Learning for Pneumonia Detection
TensorFlow/Keras implementation
"""

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

class SimpleResNet:
    """Simple ResNet50 transfer learning for pneumonia detection."""
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
    
    def build_model(self):
        """Build ResNet50 transfer learning model."""
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
    
    def compile_model(self, learning_rate=0.0001):
        """Compile the model with lower learning rate."""
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, train_generator, val_generator, epochs=20):
        """Train the model."""
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ModelCheckpoint('results/resnet_best.h5', monitor='val_accuracy', save_best_only=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7)
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

def train_resnet():
    """Train ResNet50 model."""
    print("="*60)
    print("RESNET50 TRANSFER LEARNING TRAINING")
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
    resnet = SimpleResNet()
    resnet.build_model()
    resnet.compile_model()
    
    print("Model Architecture:")
    resnet.model.summary()
    
    # Train model
    print("\nTraining ResNet50...")
    history = resnet.train(train_generator, val_generator, epochs=20)
    
    # Evaluate
    results = resnet.evaluate(val_generator)
    print(f"\nResults: Accuracy = {results['accuracy']:.4f}")
    
    return resnet, results

if __name__ == "__main__":
    train_resnet()
