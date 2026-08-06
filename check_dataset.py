import os
from pathlib import Path

# Sets the BASE_PATH to exactly where this script is located
BASE_PATH = Path(__file__).resolve().parent

def inspect_dataset():
    print(f"Verifying folders in {BASE_PATH}...\n")
    print("Performing a deep recursive search for images... (This might take a few seconds)\n")
    
    # Get all folders in the current directory, ignoring 'venv' and hidden folders
    art_styles = [
        d for d in os.listdir(BASE_PATH) 
        if os.path.isdir(BASE_PATH / d) and d != "venv" and not d.startswith('.')
    ]
    
    if len(art_styles) == 0:
        print("Error: No art style folders found!")
        return
    
    print(f"Total Art Style Folders Found: {len(art_styles)}\n")
    
    total_images = 0
    valid_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'}
    
    for style in sorted(art_styles):
        style_path = BASE_PATH / style
        
        # RECURSIVE SEARCH: .rglob('*') looks inside ALL sub-folders automatically
        images = [
            f for f in style_path.rglob('*') 
            if f.is_file() and f.suffix.lower() in valid_extensions
        ]
        
        print(f" - {style}: {len(images)} images")
        total_images += len(images)
        
    print(f"\nTotal valid images found across all subfolders: {total_images}")

if __name__ == "__main__":
    inspect_dataset()