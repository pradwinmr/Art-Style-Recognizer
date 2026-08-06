import torch
import torch.nn as nn
from torchvision import models
import matplotlib
matplotlib.use('Agg') # <-- THIS BYPASSES THE TKINTER ERROR
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from dataset import get_data_loaders # Reusing your existing data pipeline!

def main():
    # 1. Setup the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load the Validation Data
    print("Loading validation dataset...")
    # We use '_' to ignore the train_loader since we only want to test the model
    _, val_loader, classes = get_data_loaders()

    # 3. Load your trained model weights
    print("Loading saved model brain...")
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes)) 
    model.load_state_dict(torch.load("art_style_model_v2.pth", map_location=device))
    model = model.to(device)
    model.eval() # Put model in evaluation mode

    # 4. Generate Predictions
    all_preds = []
    all_labels = []

    print("Evaluating model... (This will take a minute or two)")
    with torch.no_grad(): # We don't need gradients for testing
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # Save what the model guessed vs what the actual answer was
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 5. Draw the Confusion Matrix
    print("Drawing the matrix!")
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)

    plt.xlabel('What the Model Guessed', fontsize=12, fontweight='bold')
    plt.ylabel('The Actual Art Style', fontsize=12, fontweight='bold')
    plt.title('Art Style Classification - Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.savefig('confusion_matrix.png', dpi=300)
    print("Success! Saved confusion matrix as 'confusion_matrix.png' in your project folder!")

# --- The Crucial Windows Multiprocessing Fix ---
if __name__ == '__main__':
    main()