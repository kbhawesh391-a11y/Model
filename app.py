import streamlit as st
import numpy as np
import joblib
import os

st.title("Breast Cancer Classification App")

# Debug: show current folder and files
st.write("Current folder:", os.getcwd())
st.write("Files in folder:", os.listdir())

# Load saved objects
try:
    scaler = joblib.load("scaler.pkl")
    pca = joblib.load("pca.pkl")
    model = joblib.load("best_model_breast_cancer.pkl")
    st.success("Loaded scaler, PCA, and model successfully!")
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# Feature names in the same order as training
feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

st.write("Enter feature values:")

# Collect user input
input_data = []
for feature in feature_names:
    val = st.number_input(f"{feature}", value=0.0)
    input_data.append(val)

# Predict on button click
if st.button("Predict"):
    try:
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)   # Only transform
        input_pca = pca.transform(input_scaled)
        prediction = model.predict(input_pca)
        st.success(f"Predicted Class: {prediction[0]} (0 = Malignant, 1 = Benign)")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

