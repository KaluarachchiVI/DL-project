"""
Vision Transformer (ViT) model for chest X-ray pneumonia classification using PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from transformers import TrainingArguments, Trainer
import numpy as np
import os
from utils.evaluation import ModelEvaluator
from utils.visualization import Visualizer
from utils.data_preprocessing import ChestXRayDataset

class VisionTransformerClassifier:
    """Vision Transformer classifier for chest X-ray images."""
    
    def __init__(self, num_classes=2, model_name='google/vit-base-patch16-224', 
                 random_seed=42, device=None):
        self.num_classes = num_classes
        self.model_name = model_name
        self.random_seed = random_seed
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.processor = None
        self.evaluator = ModelEvaluator()
        self.visualizer = Visualizer()
        
        # Set random seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        print(f"Using device: {self.device}")
    
    def build_model(self, num_labels=None):
        """Build the Vision Transformer model."""
        if num_labels is None:
            num_labels = self.num_classes
        
        # Load pre-trained ViT model
        self.model = ViTForImageClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        
        # Load processor
        self.processor = ViTImageProcessor.from_pretrained(self.model_name)
        
        # Move model to device
        self.model.to(self.device)
        
        print(f"Vision Transformer model loaded: {self.model_name}")
        return self.model
    
    def get_transforms(self):
        """Get image transforms for ViT."""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        return transform
    
    def train(self, train_loader, val_loader, epochs=10, learning_rate=2e-5,
              weight_decay=0.01, warmup_steps=500, save_steps=1000):
        """Train the Vision Transformer model."""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir='./models/saved/vit_output',
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir='./logs',
            logging_steps=100,
            save_steps=save_steps,
            evaluation_strategy="steps",
            eval_steps=save_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            report_to=None,  # Disable wandb
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_loader.dataset,
            eval_dataset=val_loader.dataset,
            compute_metrics=self._compute_metrics,
        )
        
        print("Training Vision Transformer model...")
        
        # Train the model
        trainer.train()
        
        print("Vision Transformer training completed!")
        return trainer
    
    def _compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = (predictions == labels).astype(np.float32).mean().item()
        return {"accuracy": accuracy}
    
    def evaluate(self, test_loader):
        """Evaluate the model."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        self.model.eval()
        y_pred_list = []
        y_true_list = []
        y_scores_list = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                probabilities = torch.softmax(outputs.logits, dim=1)
                predictions = torch.argmax(outputs.logits, dim=1)
                
                y_pred_list.extend(predictions.cpu().numpy())
                y_true_list.extend(labels.cpu().numpy())
                y_scores_list.extend(probabilities[:, 1].cpu().numpy())
        
        # Evaluate
        results = self.evaluator.evaluate_binary_classification(
            y_true_list, y_pred_list, y_scores_list, "Vision Transformer"
        )
        
        return results
    
    def predict(self, test_loader):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                probabilities = torch.softmax(outputs.logits, dim=1)
                predictions.extend(probabilities.cpu().numpy())
        
        return np.array(predictions)
    
    def save_model(self, filepath="models/saved/vision_transformer_model"):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save_pretrained(filepath)
        self.processor.save_pretrained(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath="models/saved/vision_transformer_model"):
        """Load a trained model."""
        self.model = ViTForImageClassification.from_pretrained(filepath)
        self.processor = ViTImageProcessor.from_pretrained(filepath)
        self.model.to(self.device)
        print(f"Model loaded from {filepath}")

class CustomViT(nn.Module):
    """Custom Vision Transformer implementation."""
    
    def __init__(self, img_size=224, patch_size=16, num_classes=2, 
                 embed_dim=768, num_heads=12, num_layers=12):
        super(CustomViT, self).__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H, W)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x += self.pos_embed
        
        # Transformer encoder
        x = x.transpose(0, 1)  # (num_patches+1, B, embed_dim)
        x = self.transformer(x)
        x = x.transpose(0, 1)  # (B, num_patches+1, embed_dim)
        
        # Classification
        x = self.norm(x)
        cls_output = x[:, 0]  # Class token
        output = self.head(cls_output)
        
        return output

class CustomViTTrainer:
    """Trainer for custom ViT implementation."""
    
    def __init__(self, model, device, num_epochs=50, learning_rate=1e-4):
        self.model = model
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
        # Move model to device
        self.model.to(device)
        
        # Set up optimizer and loss
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
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
    
    def train(self, train_loader, val_loader):
        """Train the model."""
        print("Training Custom Vision Transformer...")
        
        for epoch in range(self.num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            print(f'Epoch [{epoch+1}/{self.num_epochs}], '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        print("Custom Vision Transformer training completed!")

def train_vision_transformer(train_loader, val_loader, test_loader, 
                            model_type='pretrained', epochs=10):
    """Train and evaluate Vision Transformer model."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_type == 'pretrained':
        # Use pre-trained ViT
        vit = VisionTransformerClassifier(device=device)
        vit.build_model()
        
        # Train model
        trainer = vit.train(train_loader, val_loader, epochs=epochs)
        
        # Evaluate model
        results = vit.evaluate(test_loader)
        
        # Save model
        vit.save_model()
        
        return vit, results, trainer
    
    else:
        # Use custom ViT implementation
        model = CustomViT()
        trainer = CustomViTTrainer(model, device, num_epochs=epochs)
        
        # Train model
        trainer.train(train_loader, val_loader)
        
        # Evaluate model
        model.eval()
        y_pred_list = []
        y_true_list = []
        y_scores_list = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                y_pred_list.extend(predictions.cpu().numpy())
                y_true_list.extend(labels.cpu().numpy())
                y_scores_list.extend(probabilities[:, 1].cpu().numpy())
        
        # Evaluate
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_binary_classification(
            y_true_list, y_pred_list, y_scores_list, "Custom Vision Transformer"
        )
        
        return model, results, trainer

if __name__ == "__main__":
    # Test the Vision Transformer
    print("Vision Transformer model implementation loaded successfully!")
