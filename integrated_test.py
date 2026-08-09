import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from dotenv import load_dotenv
from google import genai

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
# Load the secure API key from your .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialize the Gemini Client
client = genai.Client(api_key=api_key)

# Device configuration (Use CPU for quick inference)
device = torch.device("cpu")

# Define the exact 13 classes your model was trained on
classes = [
    'Academic_Art', 'Art_Nouveau', 'Baroque', 'Expressionism', 
    'Japanese_Art', 'Neoclassicism', 'Primitivism', 'Realism', 
    'Renaissance', 'Rococo', 'Romanticism', 'Symbolism', 'Western_Medieval'
]

# ==========================================
# 2. THE CNN PREDICTION ENGINE
# ==========================================
def predict_style(image_path):
    print("Loading AI Vision Model...")
    
    # Load the ResNet-50 skeleton
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 13)
    
    # Load YOUR upgraded, trained brain
    model.load_state_dict(torch.load("art_style_model_v2.pth", map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # Prep the image for the CNN
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Make the prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    predicted_style = classes[predicted_idx.item()]
    return predicted_style, confidence.item() * 100

# ==========================================
# 3. THE LLM DESCRIPTION ENGINE
# ==========================================
def get_art_details(image_path, predicted_style):
    print(f"Contacting Gemini for historical details on this {predicted_style} piece...")
    
    # Open the raw image for Gemini
    pil_image = Image.open(image_path)
    
    # The strict background prompt
    prompt = f"""
    This painting has been classified as {predicted_style}. 
    Analyze this specific image and return EXACTLY three things in this format:
    1. Artist: [Name]
    2. Time Period: [Year or Era]
    3. Description: [A concise 3-4 line educational summary of the art]
    """
    
    # Send to the latest Gemini multimodal infrastructure
    response = client.models.generate_content(
        model='gemini-3.6-flash',
        contents=[prompt, pil_image]
    )
    
    return response.text

# ==========================================
# 4. RUN THE INTEGRATED PIPELINE
# ==========================================
if __name__ == "__main__":
    # --- Put your test image file name here ---
    test_image = "test_image.jpg.webp" 
    
    if not os.path.exists(test_image):
        print(f"Error: Could not find '{test_image}'. Please check the file name.")
    else:
        print("\n--- STARTING HYBRID AI PIPELINE ---")
        
        # Phase 1: CNN predicts the style
        style, conf = predict_style(test_image)
        print(f"\n[CNN OUTPUT] Style: {style} (Confidence: {conf:.2f}%)")
        
        # Phase 2: LLM extracts the history based on the image and CNN prediction
        details = get_art_details(test_image, style)
        
        print("\n[LLM OUTPUT]")
        print(details)
        print("\n--- PIPELINE COMPLETE ---")