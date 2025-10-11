"""
Simple Vision Transformer for Pneumonia Detection
PyTorch implementation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import vit_b_16
import numpy as np

class SimpleViT(nn.Module):
    """Simple Vision Transformer for pneumonia detection."""
    
    def __init__(self, num_classes=2):
        super(SimpleViT, self).__init__()
        
        # Load pre-trained ViT
        self.vit = vit_b_16(pretrained=True)
        
        # Replace classifier - get the input features from the last layer
        in_features = self.vit.heads[-1].in_features
        self.vit.heads = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.vit(x)

class SimpleViTTrainer:
    """Simple trainer for ViT."""
    
    def __init__(self, model, device, learning_rate=1e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        return total_loss / len(train_loader), correct / total
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return total_loss / len(val_loader), correct / total
    
    def train(self, train_loader, val_loader, epochs=10):
        """Train the model."""
        print("Training Vision Transformer...")
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        return val_acc

def train_vision_transformer():
    """Train Vision Transformer model."""
    print("="*60)
    print("VISION TRANSFORMER TRAINING")
    print("="*60)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Import data preprocessing
    import sys
    sys.path.append('..')
    from utils.data_preprocessing import DataPreprocessor, ChestXRayDataset
    
    # Load data
    preprocessor = DataPreprocessor("../data", (224, 224), 8, 42)  # Smaller batch size for ViT
    image_paths, labels = preprocessor.load_data_paths()
    X_train, X_val, y_train, y_val = preprocessor.create_train_val_split(image_paths, labels)
    
    # Create data loaders
    train_transform, val_transform = preprocessor.get_pytorch_transforms()
    
    train_dataset = ChestXRayDataset(X_train, y_train, train_transform)
    val_dataset = ChestXRayDataset(X_val, y_val, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    # Create and train model
    model = SimpleViT()
    trainer = SimpleViTTrainer(model, device)
    
    print("Model Architecture:")
    print(model)
    
    # Train model
    print("\nTraining Vision Transformer...")
    final_acc = trainer.train(train_loader, val_loader, epochs=5)  # Reduced for demo
    
    print(f"\nResults: Final Accuracy = {final_acc:.4f}")
    
    return model, final_acc

if __name__ == "__main__":
    train_vision_transformer()
