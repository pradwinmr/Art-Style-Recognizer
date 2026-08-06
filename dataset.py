import os
from pathlib import Path
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# Define the base path dynamically
BASE_PATH = Path(__file__).resolve().parent

# NEW: Point specifically to the 'data' folder
DATA_PATH = BASE_PATH / "data"

# Define standard dimensions for CNNs (e.g., ResNet or EfficientNet)
IMAGE_SIZE = 224
BATCH_SIZE = 32

def get_data_loaders():
    print("Setting up data transformations...")
    
    # 1. Define Data Augmentations for Training
    # This artificially expands our dataset by slightly altering images
    train_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5), # 50% chance to flip the image
        transforms.RandomRotation(15),          # Rotate up to 15 degrees
        transforms.ToTensor(),                  # Convert image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], # Standard ImageNet normalization
                             std=[0.229, 0.224, 0.225])
    ])

    # 2. Define Transforms for Validation/Testing (NO random flips here)
    val_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    print(f"Loading images from {DATA_PATH}...")
    
    # 3. Load the entire dataset using ImageFolder
    # Notice that 'root' is now set to DATA_PATH instead of BASE_PATH
    full_dataset = datasets.ImageFolder(root=DATA_PATH, transform=train_transforms)
    
    # 4. Split the dataset: 80% for training, 20% for validation
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Update validation dataset to use the val_transforms (no random augmentations)
    val_dataset.dataset.transform = val_transforms

    # 5. Create DataLoaders to feed data in batches
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Extract the class names (e.g., ['Academic_Art', 'Art_Nouveau', ...])
    class_names = full_dataset.classes
    
    print(f"Successfully loaded {total_size} images across {len(class_names)} classes.")
    print(f"Training set: {train_size} images | Validation set: {val_size} images")
    
    return train_loader, val_loader, class_names

if __name__ == "__main__":
    # Test the data loaders
    train_loader, val_loader, classes = get_data_loaders()
    print("\nClass mapping:", classes)