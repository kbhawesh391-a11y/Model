import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.title("Breast Cancer Classification App")
st.write("Enter features to predict breast cancer class:")

# Load trained model
model = joblib.load('best_model_breast_cancer.pkl')

# Feature names (30 features of breast cancer dataset)
feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error',
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness',
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

# Input fields
input_data = []
for feature in feature_names:
    val = st.number_input(f"{feature}", value=0.0)
    input_data.append(val)

# Prediction
if st.button("Predict"):
    input_array = np.array(input_data).reshape(1, -1)
    
    # Scale input (note: for accurate prediction, use same scaler used during training)
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_array)
    
    # PCA transformation
    pca = PCA(n_components=0.95)
    input_pca = pca.fit_transform(input_scaled)
    
    prediction = model.predict(input_pca)
    st.success(f"Predicted Class: {prediction[0]} (0 = Malignant, 1 = Benign)")
