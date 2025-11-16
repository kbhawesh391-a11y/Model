import streamlit as st
import numpy as np
import joblib

st.title("Breast Cancer Classification App")
st.write("Enter the features to predict breast cancer class:")

# Load saved objects
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
model = joblib.load("best_model_breast_cancer.pkl")

# Feature names
feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

# User input
input_data = []
for feature in feature_names:
    val = st.number_input(f"{feature}", value=0.0)
    input_data.append(val)

# Prediction
if st.button("Predict"):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    input_pca = pca.transform(input_scaled)
    prediction = model.predict(input_pca)
    st.success(f"Predicted Class: {prediction[0]} (0 = Malignant, 1 = Benign)")
