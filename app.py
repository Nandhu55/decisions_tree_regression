
import streamlit as st
import numpy as np
import pickle

# Load trained model
with open('decision_tree_regressor.pkl', 'rb') as f:
    model = pickle.load(f)

st.title('Decision Tree Regression App')
st.write('Predict Diabetes Progression using Decision Tree Regressor')

st.header('Enter Input Features')

# Create 10 input fields
feature_names = [
    "Age",
    "Sex",
    "Body Mass Index (BMI)",
    "Average Blood Pressure",
    "Total Serum Cholesterol",
    "Low-Density Lipoproteins (LDL)",
    "High-Density Lipoproteins (HDL)",
    "Total Cholesterol / HDL Ratio",
    "Serum Triglycerides Level",
    "Blood Sugar Level"
]

features = []

for feature in feature_names:
    value = st.number_input(feature, value=0.0)
    features.append(value)

# Predict button
if st.button('Predict'):
    input_data = np.array([features])
    prediction = model.predict(input_data)

    st.success(f'Predicted Value: {prediction[0]}')
