"""
GPU-Optimized Vision Transformer for Pneumonia Detection
PyTorch with proper GPU configuration
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import vit_b_16
import numpy as np
import time

class GPUOptimizedViT(nn.Module):
    """GPU-optimized Vision Transformer for pneumonia detection."""
    
    def __init__(self, num_classes=2):
        super(GPUOptimizedViT, self).__init__()
        
        # Load pre-trained ViT
        self.vit = vit_b_16(pretrained=True)
        
        # Replace classifier
        self.vit.heads = nn.Linear(self.vit.heads.in_features, num_classes)
    
    def forward(self, x):
        return self.vit(x)

class GPUOptimizedViTTrainer:
    """GPU-optimized trainer for ViT."""
    
    def __init__(self, model, device, learning_rate=1e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Enable mixed precision for faster training
        self.scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    def train_epoch(self, train_loader):
        """Train for one epoch with GPU optimization."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Use mixed precision if available
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        return total_loss / len(train_loader), correct / total
    
    def validate(self, val_loader):
        """Validate the model with GPU optimization."""
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
    
    def train(self, train_loader, val_loader, epochs=5):
        """Train the model with GPU acceleration."""
        print("🚀 Training Vision Transformer with GPU acceleration...")
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            epoch_time = time.time() - epoch_start
            
            print(f'Epoch [{epoch+1}/{epochs}] - '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
                  f'Time: {epoch_time:.2f}s')
        
        total_time = time.time() - start_time
        print(f"⏱️ Total training time: {total_time:.2f} seconds")
        
        return val_acc

def train_gpu_vit():
    """Train GPU-optimized Vision Transformer model."""
    print("="*60)
    print("GPU-OPTIMIZED VISION TRANSFORMER TRAINING")
    print("="*60)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Import data preprocessing
    import sys
    sys.path.append('..')
    from utils.data_preprocessing import DataPreprocessor, ChestXRayDataset
    
    # Load data
    preprocessor = DataPreprocessor("../data", (224, 224), 16, 42)  # Larger batch size for GPU
    image_paths, labels = preprocessor.load_data_paths()
    X_train, X_val, y_train, y_val = preprocessor.create_train_val_split(image_paths, labels)
    
    # Create data loaders
    train_transform, val_transform = preprocessor.get_pytorch_transforms()
    
    train_dataset = ChestXRayDataset(X_train, y_train, train_transform)
    val_dataset = ChestXRayDataset(X_val, y_val, val_transform)
    
    # Use larger batch size for GPU
    batch_size = 16 if device.type == 'cuda' else 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Create and train model
    model = GPUOptimizedViT()
    trainer = GPUOptimizedViTTrainer(model, device)
    
    print("Model Architecture:")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\n🚀 Training GPU-Optimized Vision Transformer...")
    final_acc = trainer.train(train_loader, val_loader, epochs=5)
    
    print(f"\nResults: Final Accuracy = {final_acc:.4f}")
    
    return model, final_acc

if __name__ == "__main__":
    train_gpu_vit()
