import torch
import torch.nn as nn
from torchvision import models

def get_art_classifier(num_classes=13):
    print("Loading pre-trained ResNet-50 vision model...")
    
    # 1. Download the pre-trained ResNet-50 model weights
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    
    # 2. Freeze the base layers so we don't destroy what it already knows
    for param in model.parameters():
        param.requires_grad = False
        
    # 3. Replace the final classification head
    # We find out how many inputs the final layer expects...
    num_ftrs = model.fc.in_features
    # ...and replace it with a new dense layer that outputs our 13 art styles
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

if __name__ == "__main__":
    # Test the model creation
    test_model = get_art_classifier(num_classes=13)
    print("\nModel Architecture Ready!")
    print(f"The final layer is properly configured to output {test_model.fc.out_features} classes.")