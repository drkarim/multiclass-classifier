import streamlit as st
import numpy as np
import pickle
import json
from PIL import Image
import cv2

# Load the model, scaler, and optimal thresholds
with open('logistic_regression_multiclass.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('optimal_thresholds.json', 'r') as thresholds_file:
    optimal_thresholds = json.load(thresholds_file)

def preprocess_image(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize to 8x8
    resized_image = cv2.resize(gray_image, (8, 8), interpolation=cv2.INTER_AREA)
    # Flatten the image to a feature vector
    feature_vector = resized_image.flatten()
    # Scale the feature vector
    scaled_vector = scaler.transform([feature_vector])
    return scaled_vector

def predict_digit(scaled_vector):
    probabilities = model.predict_proba(scaled_vector)[0]
    predictions = (probabilities >= np.array(list(optimal_thresholds.values()))).astype(int)
    predicted_digit = np.argmax(predictions)
    return predicted_digit

st.title("Digit Classifier Sketch App")
st.write("Sketch a digit in the box below and press 'Predict'.")

# Create a canvas for sketching
canvas_result = st.canvas(
    fill_color="white",
    stroke_color="black",
    stroke_width=10,
    height=200,
    width=200,
    drawing_mode='freedraw',
    key="canvas"
)

if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Convert the canvas image to a format suitable for processing
        image = np.array(canvas_result.image_data)
        scaled_vector = preprocess_image(image)
        predicted_digit = predict_digit(scaled_vector)
        st.write(f"The predicted digit is: {predicted_digit}")
        # Display the 8x8 image
        st.image(image, caption='Your Sketch', use_column_width=True)
    else:
        st.write("Please sketch a digit before predicting.")

st.write("You can re-enter another number by sketching again.")