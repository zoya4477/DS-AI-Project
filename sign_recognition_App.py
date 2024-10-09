import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('sign_language_model')

# Define a function to make predictions
def predict_image(image):
    # Resize the image to 224x224 and convert to RGB
    img = image.resize((224, 224))
    img = np.array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict using the model
    predictions = model.predict(img)
    predicted_label = np.argmax(predictions, axis=1)
    return predicted_label

# Streamlit interface
st.title("Sign Language Recognition")
st.header("Upload an image of a sign language gesture")

# File uploader for images
uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make prediction
    st.write("Classifying...")
    label = predict_image(image)

    # Display the prediction result
    st.write(f"The predicted sign language alphabet is: {chr(label[0] + 65)}")  # Convert to corresponding letter
