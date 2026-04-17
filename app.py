import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from PIL import Image
import os

# --------------------------------------------------
# Helper: Load model safely
# --------------------------------------------------
@st.cache_resource
def load_trained_model():
    model_path = os.path.join(os.path.dirname(__file__), "best_vgg16_skin_model.h5")

    if not os.path.exists(model_path):
        st.warning(
            f"⚠️ Model file not found at: `{model_path}`\n\n"
            "Please make sure `best_vgg16_skin_model.h5` is in the same directory as `app.py`."
        )
        return None

    return load_model(model_path)

# Load model once
model = load_trained_model()

# --------------------------------------------------
# VGG16 Base (for consistency)
# --------------------------------------------------
# The base model expects input_shape = (224, 224, 3)
vgg16 = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Class labels
class_labels = [
    'Acne and Rosacea',
    'Actinic Keratosis',
    'Eczema',
    'Melanoma',
    'Psoriasis'
]

# --------------------------------------------------
# Image Preprocessing Function
# --------------------------------------------------
def preprocess_image(image):
    """Resize and preprocess the image for VGG16."""
    image = image.resize((224, 224))  # ✅ Fix: match VGG16 input size
    image = np.array(image)

    # Handle alpha channel if present
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[:, :, :3]

    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("🩺 Skin Disease Detection Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Skin Disease Detection"])

# --------------------------------------------------
# Home Page
# --------------------------------------------------
if app_mode == "Home":
    st.title("🩺 SKIN DISEASE DETECTION")
    st.write("Welcome to the **Skin Disease Detection App**.")
    st.write("Use the sidebar to navigate between pages.")

# --------------------------------------------------
# About Page
# --------------------------------------------------
elif app_mode == "About":
    st.title("ℹ️ About")
    st.write("""
    This application uses a deep learning model trained on dermatological images
    to detect various skin diseases from uploaded photos.
    
    **Model:** VGG16 (feature extraction)  
    **Framework:** TensorFlow / Keras  
    **Authors:** [K.Venkata Ravindra Reddy / D.Sudheer Kumar]
    """)

# --------------------------------------------------
# Skin Disease Detection Page
# --------------------------------------------------
elif app_mode == "Skin Disease Detection":
    st.title("🔬 Skin Disease Detection")
    st.write("Upload an image of skin to detect potential disease.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if model is None:
            st.error("❌ The model file is missing. Please upload `best_vgg16_skin_model.h5` and restart the app.")
        else:
            # Preprocess the image
            processed_image = preprocess_image(image)

            # Make prediction
            predictions = model.predict(processed_image)
            predicted_class = np.argmax(predictions)
            predicted_label = class_labels[predicted_class]

            # Display results
            st.subheader(f"✅ Prediction: **{predicted_label}**")
            st.write("### Prediction Probabilities:")
            for i, label in enumerate(class_labels):
                st.write(f"{label}: {predictions[0][i]:.4f}")
