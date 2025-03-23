# QR Code Authentication - Deep Learning Pipeline with Streamlit Web App

## Section 1: Load Dataset
import os
import cv2
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DATA_DIR = "./../Data/"
CATEGORIES = ["first_print", "second_print"]  # Adjust category names if needed
IMG_SIZE = 128  # Resize images to 128x128

## Section 2: Load Pretrained Model
MODEL_PATH = "Notebook/qr_code_authentication_model.h5"
if os.path.exists(MODEL_PATH):
    model = keras.models.load_model(MODEL_PATH)
    st.success("✅ Model loaded successfully!")
else:
    st.error("❌ Error: Model file not found!")

## Section 3: Streamlit Web App
st.title("🔍 QR Code Authentication System")
st.write("📌 Upload a QR code image to verify if it is original or counterfeit.")

uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "png", "jpeg"])

def predict_qr_code(uploaded_file, model):
    
    img_size=IMG_SIZE

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        if image is None:
            st.error("❌ Error: Unable to process the uploaded image.")
            return
        image = cv2.resize(image, (img_size, img_size))
        image = image.reshape(1, img_size, img_size, 1) / 255.0  # Normalize
        prediction = model.predict(image)[0][0]
        label = "✅ First Print (Original)" if prediction < 0.5 else "⚠️ Second Print (Counterfeit)"
        
        # Displaying results in a highlighted manner
        st.subheader("🔎 Prediction Result")
        st.markdown(f"### **{label}**")
        
        # Display uploaded image
        st.image(image, caption="Uploaded QR Code", use_column_width=True, clamp=True)

if uploaded_file and 'model' in locals():
    predict_qr_code(uploaded_file, model)
