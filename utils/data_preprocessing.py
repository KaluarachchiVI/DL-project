"""
Data preprocessing utilities for chest X-ray pneumonia classification.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import torch

# Configure TensorFlow GPU
def setup_tensorflow_gpu():
    """Configure TensorFlow to use GPU properly."""
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"TensorFlow using GPU: {gpus[0]}")
            return True
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
            return False
    return False

# Initialize GPU configuration
setup_tensorflow_gpu()
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import warnings
warnings.filterwarnings('ignore')

class ChestXRayDataset(Dataset):
    """PyTorch Dataset for chest X-ray images."""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class DataPreprocessor:
    """Data preprocessing class for chest X-ray images."""
    
    def __init__(self, data_dir, img_size=(224, 224), batch_size=32, random_seed=42):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.random_seed = random_seed
        
        # Set random seeds for reproducibility
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Initialize label encoder
        self.label_encoder = LabelEncoder()
        
    def load_data_paths(self):
        """Load image paths and labels from directory structure."""
        image_paths = []
        labels = []
        
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(self.data_dir, split)
            if not os.path.exists(split_dir):
                print(f"Warning: {split_dir} does not exist. Skipping...")
                continue
                
            for class_name in ['NORMAL', 'PNEUMONIA']:
                class_dir = os.path.join(split_dir, class_name)
                if not os.path.exists(class_dir):
                    print(f"Warning: {class_dir} does not exist. Skipping...")
                    continue
                    
                for filename in os.listdir(class_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(class_dir, filename))
                        labels.append(class_name)
        
        return image_paths, labels
    
    def create_train_val_split(self, image_paths, labels, val_split=0.2):
        """Create train/validation split from all data."""
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            image_paths, encoded_labels, 
            test_size=val_split, 
            random_state=self.random_seed,
            stratify=encoded_labels
        )
        
        return X_train, X_val, y_train, y_val
    
    def get_tensorflow_generators(self, X_train, X_val, y_train, y_val, augmentation=True):
        """Create TensorFlow data generators."""
        
        # Data augmentation for training
        if augmentation:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                shear_range=0.2,
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)
        
        # No augmentation for validation
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            self.data_dir + '/train',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            seed=self.random_seed
        )
        
        val_generator = val_datagen.flow_from_directory(
            self.data_dir + '/val',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            seed=self.random_seed
        )
        
        return train_generator, val_generator
    
    def get_pytorch_transforms(self, augmentation=True):
        """Get PyTorch transforms for data augmentation."""
        
        if augmentation:
            train_transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.1),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
            ])
        
        val_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform
    
    def get_pytorch_dataloaders(self, X_train, X_val, y_train, y_val, num_workers=4):
        """Create PyTorch data loaders."""
        
        train_transform, val_transform = self.get_pytorch_transforms()
        
        # Create datasets
        train_dataset = ChestXRayDataset(X_train, y_train, train_transform)
        val_dataset = ChestXRayDataset(X_val, y_val, val_transform)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def preprocess_for_sklearn(self, image_paths, labels):
        """Preprocess images for scikit-learn models (flattened features)."""
        processed_images = []
        
        for image_path in image_paths:
            # Load and resize image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.img_size)
            
            # Flatten image
            flattened = image.flatten()
            processed_images.append(flattened)
        
        return np.array(processed_images), np.array(labels)
    
    def get_class_weights(self, y_train):
        """Calculate class weights for imbalanced dataset."""
        from sklearn.utils.class_weight import compute_class_weight
        
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        
        return dict(zip(np.unique(y_train), class_weights))

def check_gpu_availability():
    """Check GPU availability for TensorFlow and PyTorch."""
    print("=== GPU Availability Check ===")
    
    # TensorFlow GPU check
    print(f"TensorFlow version: {tf.__version__}")
    print(f"TensorFlow GPU available: {tf.config.list_physical_devices('GPU')}")
    
    # PyTorch GPU check
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"PyTorch CUDA device count: {torch.cuda.device_count()}")
        print(f"PyTorch current device: {torch.cuda.current_device()}")
    
    print("=" * 30)

if __name__ == "__main__":
    # Test the data preprocessor
    preprocessor = DataPreprocessor("data")
    check_gpu_availability()
    
    # Load data paths
    image_paths, labels = preprocessor.load_data_paths()
    print(f"Total images found: {len(image_paths)}")
    print(f"Labels: {set(labels)}")
