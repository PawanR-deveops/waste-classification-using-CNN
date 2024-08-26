import numpy as np
import cv2
import os
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import tempfile

# Define the paths
train_path = "TRAIN"
test_path = "TEST"
model_path = "my_model.h5"

# Load the pre-trained model
model = load_model(model_path)

# Define the class labels
class_labels = ['Organic', 'Recyclable']

# Streamlit web app
st.title('Waste Classification App')

# Define a function to make predictions
def predict_waste(image):
    img = cv2.resize(image, (224, 224))
    img = np.reshape(img, [-1, 224, 224, 3])
    result = np.argmax(model.predict(img))
    return class_labels[result]

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_image.read())
        temp_file_path = temp_file.name

    image = cv2.imread(temp_file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Classify'):
        prediction = predict_waste(image)
        st.success(f'This image is classified as: {prediction}')
        
    # Clean up the temporary file
    os.unlink(temp_file_path)