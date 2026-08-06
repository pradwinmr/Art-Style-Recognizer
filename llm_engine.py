import os
from dotenv import load_dotenv
from google import genai
from PIL import Image

# 1. Load the hidden API key from your .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("Error: Could not find GEMINI_API_KEY in the .env file.")
    exit()

# 2. Initialize the NEW Google GenAI client
client = genai.Client(api_key=GEMINI_API_KEY)

def get_art_description(image_path, predicted_style):
    print("Contacting Gemini for artwork details...")
    
    try:
        # Load the image
        img = Image.open(image_path)
    except FileNotFoundError:
        return f"Error: Could not find the image at {image_path}. Please make sure the file exists."
    
    # Write the strict background prompt
    prompt = f"""
    This image has been classified by a computer vision model as {predicted_style}. 
    Analyze the image and provide exactly three things formatted clearly:
    1. Artist: (Name of the specific creator)
    2. Time Period: (The specific era, decade, or year)
    3. Description: (A concise 3-4 line educational summary of the art)
    
    Do not include any conversational filler. Just return the requested facts.
    """
    
    # 3. Send the prompt and the image using the new syntax and the 2.5-flash model
    response = client.models.generate_content(
        model='gemini-3.6-flash',
        contents=[prompt, img]
    )
    
    return response.text

# --- Quick Test ---
if __name__ == "__main__":
    # Pointing to the Rembrandt image you successfully predicted earlier
    test_image = "test_image.jpg.webp" 
    test_style = "Baroque"
    
    try:
        details = get_art_description(test_image, test_style)
        print("\n--- Gemini Output ---")
        print(details)
    except Exception as e:
        print(f"An error occurred: {e}")