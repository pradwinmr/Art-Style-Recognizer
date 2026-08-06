import streamlit as st
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from dotenv import load_dotenv
from google import genai
import time

# ==========================================
# 1. PAGE CONFIGURATION & STATE
# ==========================================
st.set_page_config(page_title="Art Style Recognizer", layout="wide", initial_sidebar_state="expanded")

# Initialize short-term memory (Session State) for the "Recent Arts" sidebar
if "history" not in st.session_state:
    st.session_state.history = []

# ==========================================
# 2. AI ENGINE INITIALIZATION
# ==========================================
# Load API Key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# The 13 trained classes
classes = [
    'Academic_Art', 'Art_Nouveau', 'Baroque', 'Expressionism', 
    'Japanese_Art', 'Neoclassicism', 'Primitivism', 'Realism', 
    'Renaissance', 'Rococo', 'Romanticism', 'Symbolism', 'Western_Medieval'
]

# Cache the heavy PyTorch model so it only loads once!
@st.cache_resource
def load_vision_model():
    device = torch.device("cpu")
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 13)
    
    # Load your highly accurate trained weights
    model.load_state_dict(torch.load("art_style_model_v2.pth", map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model, device

vision_model, computation_device = load_vision_model()

# ==========================================
# 3. CORE PROCESSING FUNCTIONS
# ==========================================
def predict_style(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image.convert("RGB")).unsqueeze(0).to(computation_device)

    with torch.no_grad():
        outputs = vision_model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    return classes[predicted_idx.item()], confidence.item() * 100

import time # Ensure this is at the very top of your file with the other imports

def get_art_details(image, predicted_style):
    # FIX 1: Shrink the image copy for the LLM so it sends lightning fast
    llm_image = image.copy()
    llm_image.thumbnail((800, 800)) 
    
    prompt = f"""
    This painting has been classified as {predicted_style}. 
    Analyze this specific image and return EXACTLY four things in this format:
    1. Title: [Name of the artwork]
    2. Artist: [Name of the artist]
    3. Time Period: [Year or Era]
    4. Description: [A concise 3-4 line educational summary of the art]
    """
    
    # FIX 2: Automated Retry Loop (Max 3 attempts)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Try to contact Gemini with the lightweight image
            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=[prompt, llm_image]
            )
            return response.text
            
        except Exception as e:
            # If the server is busy, check if we have retries left
            if attempt < max_retries - 1:
                time.sleep(2) # Silently wait 2 seconds, then loop back and try again
                continue
            else:
                # If it fails 3 times in a row, THEN show the UI warning
                return f"**⚠️ AI Engine Overloaded:** The Gemini servers are currently experiencing peak global demand and couldn't fetch the historical details right now. \n\n*Please wait a minute or two and try uploading the artwork again!*"
            
# ==========================================
# 4. SIDEBAR UI (Left Panel)
# ==========================================
with st.sidebar:
    st.header("🕰️ Recent Arts")
    
    if len(st.session_state.history) == 0:
        st.write("*None*")
    else:
        # Display history in reverse order (newest first)
        for past_art in reversed(st.session_state.history):
            st.image(past_art["image"], caption=f"{past_art['style']} ({past_art['confidence']:.1f}%)")
            st.divider()
            
    # Push system info to the absolute bottom using empty space
    st.markdown("<br>" * 10, unsafe_allow_html=True)
    st.caption("⚙️ **System Info:**\nVision Model: ResNet-50 (Fine-Tuned on RTX 3050)\nLLM Engine: Gemini 3.6 Flash")
    st.caption("🔗 [GitHub Repository](#)")

# ==========================================
# 5. MAIN SCREEN UI (Center)
# ==========================================
st.markdown("<h1 style='text-align: center;'>Art Style Recognizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an artwork to reveal its Art style, Time-Period, Artist, and Description.</p>", unsafe_allow_html=True)

# The centralized Drag-and-Drop Uploader
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Open the image from the uploaded buffer
    pil_image = Image.open(uploaded_file)
    
    # Visual processing cue
    with st.spinner("Analyzing brushstrokes and consulting historical records..."):
        # 1. Run CNN
        style, conf = predict_style(pil_image)
        # 2. Run LLM
        details = get_art_details(pil_image, style)
        
        # Save to short-term memory (Recent Arts)
        # NEW: Check the filename to prevent duplicates in the sidebar when hitting Reupload!
        is_duplicate = len(st.session_state.history) > 0 and st.session_state.history[-1].get("filename") == uploaded_file.name
        
        if not is_duplicate:
            st.session_state.history.append({
                "image": pil_image,
                "style": style,
                "confidence": conf,
                "filename": uploaded_file.name
            })
            # --- PHASE 4 FIX: Clear the old chat history for the new artwork ---
            st.session_state.messages = []

    # The Split-Column Output
    st.markdown("---")
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        # Display the image nicely scaled
        st.image(pil_image, use_container_width=True)
        
    with col2:
        # Display the CNN Math with Confidence Threshold
        style_clean = style.replace('_', ' ')
        
        if conf >= 50:
            st.subheader(f"Style: **{style_clean}**")
            st.progress(int(conf), text=f"CNN Confidence: {conf:.2f}%")
        else:
            st.subheader(f"Style: **{style_clean}** *(Low Confidence)*")
            st.progress(int(conf), text=f"CNN Confidence: {conf:.2f}%")
            st.info(
                "💡 **Ambiguous / Hybrid Visual Style Detected:**\n"
                f"The vision model predicted **{style_clean}** with low confidence ({conf:.1f}%). "
                "This painting likely sits on the boundary between art movements or features visual techniques from multiple styles."
            )
        
        st.markdown("---")
        
        # NEW: The Reupload Button Logic
        if "⚠️ AI Engine Overloaded" in details:
            st.warning(details)
            # This button instantly reruns the script using the currently uploaded image
            st.button("🔄 Reupload") 
        else:
            # If successful, just print the normal details
            st.markdown(details)

# --- PHASE 3: INTERACTIVE Q&A CHATBOT (MOVED TO RIGHT COLUMN) ---
        st.markdown("---")
        st.subheader("💬 Ask the AI Curator")
        
        # Initialize the chat history in Streamlit's short-term memory
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display any previous chat messages in this right column
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # The Chat Input Box
        if user_question := st.chat_input("Ask a quick question..."):
            
            # 1. Display the user's question
            with st.chat_message("user"):
                st.markdown(user_question)
            
            st.session_state.messages.append({"role": "user", "content": user_question})

            # 2. Get the AI's response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    
                    # (NEW PROMPT RULE) Forcing concise, fast answers
                    chat_prompt = f"""
                    You are an expert art historian. The uploaded image is classified as {style}. 
                    The user is asking a question about it. 
                    RULE: Keep your answer incredibly concise (1 to 2 sentences max) unless the user explicitly asks for a detailed explanation.
                    Question: {user_question}
                    """
                    
                    try:
                        chat_response = client.models.generate_content(
                            model='gemini-3.6-flash',
                            contents=[chat_prompt, pil_image]
                        )
                        reply = chat_response.text
                    except Exception as e:
                        reply = "⚠️ The server is currently busy. Please try asking your question again in a moment."
                    
                    st.markdown(reply)
            
            # 3. Save the AI's response to memory
            st.session_state.messages.append({"role": "assistant", "content": reply})