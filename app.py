import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Fuel Efficiency Predictor")

st.title("🚗 Vehicle Fuel Efficiency Prediction (MPG)")

st.write("Enter vehicle details:")

# -------------------------------
# LOAD MODEL + SCALER
# -------------------------------

model = load_model(
    "fuel_efficiency_tuned_ann.keras",
    compile=False
)


scaler = joblib.load("scaler.pkl")

# -------------------------------
# USER INPUTS
# -------------------------------

year = st.number_input("Year", 1990, 2025, 2015)
hp = st.number_input("Engine HP", 50, 1000, 150)
cyl = st.number_input("Engine Cylinders", 2, 16, 4)
doors = st.number_input("Number of Doors", 2, 5, 4)
popularity = st.number_input("Popularity", 0, 6000, 1000)

fuel_type = st.selectbox(
    "Engine Fuel Type",
    ["regular unleaded", "premium unleaded", "diesel", "electric"]
)

transmission = st.selectbox(
    "Transmission Type",
    ["AUTOMATIC", "MANUAL"]
)

driven_wheels = st.selectbox(
    "Driven Wheels",
    ["front wheel drive", "rear wheel drive", "all wheel drive"]
)

vehicle_size = st.selectbox(
    "Vehicle Size",
    ["Compact", "Midsize", "Large"]
)

vehicle_style = st.selectbox(
    "Vehicle Style",
    ["Sedan", "SUV", "Coupe", "Hatchback", "Convertible", "Wagon"]
)

# -------------------------------
# CREATE INPUT DATAFRAME
# -------------------------------

input_dict = {
    "Year": year,
    "Engine HP": hp,
    "Engine Cylinders": cyl,
    "Number of Doors": doors,
    "Popularity": popularity,
    "Engine Fuel Type": fuel_type,
    "Transmission Type": transmission,
    "Driven_Wheels": driven_wheels,
    "Vehicle Size": vehicle_size,
    "Vehicle Style": vehicle_style
}

input_df = pd.DataFrame([input_dict])

# -------------------------------
# ONE HOT ENCODING
# -------------------------------

input_encoded = pd.get_dummies(input_df)

# -------------------------------
# ALIGN WITH TRAINING FEATURES
# -------------------------------

training_cols = joblib.load("training_columns.pkl")

for col in training_cols:
    if col not in input_encoded:
        input_encoded[col] = 0

input_encoded = input_encoded[training_cols]

# -------------------------------
# SCALE
# -------------------------------

input_scaled = scaler.transform(input_encoded)

# -------------------------------
# PREDICTION
# -------------------------------

if st.button("Predict Fuel Efficiency"):

    prediction = model.predict(input_scaled)

    st.success(
        f"🚀 Predicted Average Fuel Efficiency: {prediction[0][0]:.2f} MPG"
    )
