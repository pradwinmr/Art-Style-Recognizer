import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from dataset import get_data_loaders
import numpy as np

def main():
    # 1. Setup the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load the Data
    train_loader, val_loader, classes = get_data_loaders()

    # 3. Calculate Class Weights (Addressing Imbalance)
    # Using the exact counts from your EDA graph
    image_counts = [1305, 3035, 5312, 2607, 2235, 3115, 1324, 5373, 6192, 2521, 6813, 1510, 1158]
    total_images = sum(image_counts)
    
    # Apply the mathematical penalty formula
    class_weights = [total_images / (len(classes) * count) for count in image_counts]
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    print("Class weights applied to penalize majority classes!")

    # 4. Load the Pre-trained Model & Unfreeze for Fine-Tuning
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # UNFREEZE the last two layers (Layer 3 and Layer 4) of ResNet
    for name, param in model.named_parameters():
        if "layer3" in name or "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Replace the head
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))
    model = model.to(device)

    # 5. Define the Loss Function with the new Weights
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # 6. Setup the Optimizer with a smaller learning rate for fine-tuning
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

    # 7. Train Longer (10 Epochs)
    num_epochs = 10
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_corrects.double() / len(val_loader.dataset)
        print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')

    # 8. Save the upgraded brain
    torch.save(model.state_dict(), 'art_style_model_v2.pth')
    print("Advanced model saved as art_style_model_v2.pth!")

if __name__ == '__main__':
    main()