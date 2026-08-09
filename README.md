# 🎨 Art Style Recognizer: Hybrid AI Architecture

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-CNN-EE4C2C.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg)
![Gemini](https://img.shields.io/badge/Gemini-3.6_Flash-8E75B2.svg)

A modern, hybrid AI web application that acts as a digital museum curator. This project bridges the gap between raw computer vision and historical context by combining a custom-trained Convolutional Neural Network (CNN) with a Multimodal Vision-Language Model (VLM).

## 🚀 How It Works

1. **The Vision Brain (CNN):** A pre-trained `ResNet-50` model, fine-tuned on a 29 GB dataset of 42,500 high-resolution paintings. It analyzes raw pixel data (brushstrokes, lighting, geometry) to classify the artwork into one of 13 historical art movements (e.g., Baroque, Cubism, Impressionism).
2. **The Historical Brain (LLM):** Google's `Gemini 3.6 Flash` acts as the safety net and historian. It instantly processes the uploaded image alongside the CNN's prediction to extract the specific Artist, Title, Time Period, and an educational summary.
3. **The Interactive Curator:** Users can utilize the built-in chat interface to ask specific, dynamic follow-up questions about the artwork's history, techniques, or hidden meanings.

## ✨ Features

* **Instant Classification:** Identifies 13 distinct art styles with high accuracy.
* **Smart Confidence Thresholds:** Automatically detects ambiguous or hybrid art styles if the CNN's mathematical confidence drops below 50%.
* **Interactive Chatbot:** A memory-enabled Q&A sidebar for deep dives into specific paintings.
* **Premium UI/UX:** Built with Streamlit, featuring an immersive blurred gallery background, animated dropzones, dynamic loading sequences, and toast notifications.
* **Robust Error Handling:** Features an automated 3-attempt retry loop to bypass LLM server overloads, complete with a manual "Reupload" UI failsafe.

## 🛠️ Tech Stack

* **Backend / Machine Learning:** PyTorch, torchvision
* **Frontend / UI:** Streamlit, Custom CSS
* **LLM Engine:** Google GenAI SDK (`gemini-3.6-flash`)
* **Hardware:** Trained locally utilizing CUDA on an NVIDIA RTX 3050.

## ⚙️ Installation & Usage

### 1. Clone the repository
```bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/Art-Style-Recognizer.git](https://github.com/YOUR_GITHUB_USERNAME/Art-Style-Recognizer.git)
cd Art-Style-Recognizer