# 🎨 Art Style Recognizer

### 1) Developer
* **Name** - Pradwin MR
* **USN** - 1NT24AD043
* **Dept** - Artificial Intelligence and Data Science 

### 2) What is Art Style Recognition?
At its core, Art Style Recognition is a computer vision task where a machine learning model analyzes a visual image of a painting or drawing and automatically determines the historical, artistic, or stylistic movement it belongs to. Instead of seeing a painting the way humans do—as a subject like "a vase of flowers" or "a person"—the model looks at the underlying visual patterns, brushstrokes, color palettes, and geometric forms. In this project, the model acts like an automated digital art curator: it inspects the canvas, identifies the artistic technique, and provides a concise snippet explaining the style and history.

### 3) Live Demo
* **Web Application Link** - [Art Style Recognizer (Streamlit)](https://art-style-recognizer-p99hkcdcc6vbdh3z4dogc9.streamlit.app/)
* **Local Application** - To run locally, clone the repo, activate your virtual environment, and run: `streamlit run app.py`
* **Repository Link** - [https://github.com/pradwinmr/Art-Style-Recognizer](https://github.com/pradwinmr/Art-Style-Recognizer)

### 4) Dataset
* **Source:** Kaggle WikiArt Dataset.
* **Size:** ~29 GB, containing 42,500 high-resolution images.
* **Classes (13 Historical Art Movements):**
  1. Academic Art (1,305 images)
  2. Art Nouveau (3,035 images)
  3. Baroque (5,312 images)
  4. Expressionism (2,607 images)
  5. Japanese Art / Ukiyo-e (2,235 images)
  6. Neoclassicism (3,115 images)
  7. Primitivism (1,324 images)
  8. Realism (5,373 images)
  9. Renaissance (6,192 images)
  10. Rococo (2,521 images)
  11. Romanticism (6,813 images)
  12. Symbolism (1,510 images)
  13. Western Medieval (1,158 images)

### 5) Accuracy & Matrix
* **Final Model Performance:** After 10 epochs of training on an NVIDIA RTX 3050, the model achieved a **Training Accuracy of 97.74%** and a highly generalized **Validation Accuracy of 77.64%**.
* **Data Rebalancing:** The dataset featured heavy class imbalances (e.g., 6,813 Romanticism images vs. 1,158 Western Medieval images). Inverse Frequency Weighting was applied to mathematically penalize the model for missing rare categories.
* **Confusion Matrix:** Generated across 8,500 unseen validation images to map the model's "blind spots." The model performed exceptionally well on structured media like Ukiyo-e (94.5%) and Cubism (84.0%), while showing understandable historical overlaps between styles like Impressionism and Post-Impressionism.

### 6) Architecture
This project utilizes a **Hybrid AI Architecture** combining a Convolutional Neural Network (CNN) with a Multimodal Vision-Language Model (LLM):
* **The Vision Brain (CNN):** A pre-trained `ResNet-50` model is used for feature extraction via Transfer Learning. The upper layers (`layer3` and `layer4`) were unfrozen for fine-tuning, and a custom classification head was attached to output 13 style probabilities.
* **The Historian (LLM):** Because the CNN only maps pixels to a general style label (e.g., "Baroque"), the app instantly passes the image and the predicted label to Google's **Gemini 3.6 Flash** via API. A strict hidden prompt forces the LLM to return the Artist, Time Period, and an educational description.
* **Interactive Chatbot:** Users can converse with the LLM in a right-hand sidebar to ask dynamic follow-up questions about the painting's symbolism and techniques. 

### 7) Tech Stack
* **Deep Learning Framework:** PyTorch & torchvision
* **Computer Vision Model:** ResNet-50 (Fine-Tuned on NVIDIA RTX 3050)
* **Multimodal LLM Engine:** Google GenAI SDK (`gemini-3.6-flash`)
* **Frontend Web Application:** Streamlit (with Custom CSS)
* **Data Processing:** Python, Pillow (PIL), Pandas, Matplotlib, Scikit-learn, Seaborn

### 8) Future Enhancements
1. **The "Art-RAG" Engine:** Creating vector embeddings of the dataset using FAISS/ChromaDB to display a "Visually Similar Artworks" gallery when a user uploads a painting.
2. **"Museum Audio Guide" (Text-to-Speech):** Integrating a TTS API (like ElevenLabs) to read the Gemini-generated historical context aloud, mimicking an authentic museum audio tour.
3. **Clickable Canvas Exploration:** Integrating object detection (YOLO or SAM) to highlight specific symbolic elements in the painting that users can click on for detailed explanations.
4. **Cloud Database & User Galleries:** Integrating Firebase/Supabase and a login system so users can save their personal "Digital Galleries" of analyzed art.
5. **Heatmap Visualizations (Grad-CAM):** Displaying a thermal heatmap overlay over the uploaded image to visually show exactly which brushstrokes the ResNet-50 neural network stared at the hardest to make its decision.

### 9) Model Weights Installation
The pre-trained PyTorch model weights are required for the computer vision pipeline to function.
1. **Standard Installation:** The fine-tuned `art_style_model_v2.pth` file is included natively in this repository. When you `git clone` the project, the weights will automatically download. Ensure this `.pth` file remains in the root directory of the project alongside `app.py`.
2. **Custom Retraining (Optional):** If you wish to retrain the CNN on your own hardware or dataset, simply run `python train.py`. The script will process the dataset and automatically overwrite and save a newly trained `art_style_model_v2.pth` file in the correct root location.