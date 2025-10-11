"""
Transfer Learning model using ResNet50 for chest X-ray pneumonia classification.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, DenseNet121, VGG16, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import os
from utils.evaluation import ModelEvaluator
from utils.visualization import Visualizer

class TransferLearningModel:
    """Transfer learning model using pre-trained architectures."""
    
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
    
    def build_model(self, base_model='resnet50', trainable_layers=None, 
                   dropout_rate=0.5, dense_units=512):
        """Build transfer learning model."""
        
        # Load pre-trained model
        if base_model == 'resnet50':
            base = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif base_model == 'densenet121':
            base = DenseNet121(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif base_model == 'vgg16':
            base = VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif base_model == 'efficientnet':
            base = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        else:
            raise ValueError("Base model must be 'resnet50', 'densenet121', 'vgg16', or 'efficientnet'")
        
        # Freeze base model layers
        base.trainable = False
        
        # Add custom classification head
        x = base.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(dense_units, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(dense_units // 2, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate / 2)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        self.model = Model(inputs=base.input, outputs=predictions)
        
        # Unfreeze specified layers for fine-tuning
        if trainable_layers is not None:
            self._unfreeze_layers(trainable_layers)
        
        return self.model
    
    def _unfreeze_layers(self, trainable_layers):
        """Unfreeze specified layers for fine-tuning."""
        if isinstance(trainable_layers, int):
            # Unfreeze last N layers
            for layer in self.model.layers[-trainable_layers:]:
                layer.trainable = True
        elif isinstance(trainable_layers, list):
            # Unfreeze specific layers
            for layer_name in trainable_layers:
                for layer in self.model.layers:
                    if layer.name == layer_name:
                        layer.trainable = True
        else:
            # Unfreeze all layers
            for layer in self.model.layers:
                layer.trainable = True
    
    def compile_model(self, learning_rate=0.001, optimizer='adam', 
                     fine_tuning=False):
        """Compile the model."""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        # Choose optimizer based on fine-tuning
        if fine_tuning:
            # Use lower learning rate for fine-tuning
            if optimizer == 'adam':
                opt = Adam(learning_rate=learning_rate * 0.1)
            else:
                opt = SGD(learning_rate=learning_rate * 0.1, momentum=0.9)
        else:
            if optimizer == 'adam':
                opt = Adam(learning_rate=learning_rate)
            else:
                opt = SGD(learning_rate=learning_rate, momentum=0.9)
        
        self.model.compile(
            optimizer=opt,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Transfer learning model compiled successfully!")
    
    def train(self, train_generator, val_generator, epochs=50, 
              fine_tuning_epochs=20, batch_size=32, callbacks=None, verbose=1):
        """Train the model with optional fine-tuning."""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        # Default callbacks
        if callbacks is None:
            callbacks = self._get_default_callbacks()
        
        print("Training Transfer Learning model...")
        
        # Phase 1: Train only the classification head
        print("Phase 1: Training classification head...")
        self.history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Phase 2: Fine-tuning (if specified)
        if fine_tuning_epochs > 0:
            print("Phase 2: Fine-tuning...")
            
            # Unfreeze some layers for fine-tuning
            self._unfreeze_layers(10)  # Unfreeze last 10 layers
            
            # Recompile with lower learning rate
            self.compile_model(learning_rate=0.0001, fine_tuning=True)
            
            # Continue training
            fine_tuning_history = self.model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=fine_tuning_epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=verbose
            )
            
            # Combine histories
            for key in self.history.history:
                self.history.history[key].extend(fine_tuning_history.history[key])
        
        print("Transfer learning training completed!")
        return self.history
    
    def _get_default_callbacks(self):
        """Get default training callbacks."""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'models/saved/resnet_transfer_best.h5',
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
            y_true, y_pred, y_pred_proba[:, 1], "ResNet50 Transfer Learning"
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
            self.history, "ResNet50 Transfer Learning"
        )
    
    def save_model(self, filepath="models/saved/resnet_transfer_model.h5"):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath="models/saved/resnet_transfer_model.h5"):
        """Load a trained model."""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def get_model_summary(self):
        """Get model summary."""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        return self.model.summary()
    
    def get_feature_maps(self, image, layer_name='conv5_block3_out'):
        """Extract feature maps from a specific layer."""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        # Create a model that outputs the specified layer
        feature_extractor = Model(
            inputs=self.model.input,
            outputs=self.model.get_layer(layer_name).output
        )
        
        # Extract features
        features = feature_extractor.predict(np.expand_dims(image, axis=0))
        return features

def train_resnet_transfer(train_generator, val_generator, test_generator,
                         base_model='resnet50', epochs=30, fine_tuning_epochs=10):
    """Train and evaluate ResNet50 transfer learning model."""
    
    # Initialize transfer learning model
    transfer_model = TransferLearningModel()
    
    # Build model
    transfer_model.build_model(base_model=base_model)
    transfer_model.compile_model()
    
    # Print model summary
    print("Model Architecture:")
    transfer_model.get_model_summary()
    
    # Train model
    history = transfer_model.train(
        train_generator, val_generator, 
        epochs=epochs, fine_tuning_epochs=fine_tuning_epochs
    )
    
    # Plot training history
    transfer_model.plot_training_history()
    
    # Evaluate model
    results = transfer_model.evaluate(test_generator)
    
    # Save model
    transfer_model.save_model()
    
    return transfer_model, results, history

if __name__ == "__main__":
    # Test the transfer learning model
    print("ResNet50 Transfer Learning model implementation loaded successfully!")
