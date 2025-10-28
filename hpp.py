# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# === Define paths ===
BASE_DIR = r"C:\Users\E105484\OneDrive - Road Accident Fund\Documents\Regynisis\StreamlitApp"
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "one_hot_encoder.pkl")
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, "feature_names.pkl")

# === Load saved objects ===
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)
feature_names = joblib.load(FEATURE_NAMES_PATH)

# === Streamlit UI ===
st.title("🚗 Used Car Price Prediction,by Brilliance Legong")

# Input fields
vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=50, value=3)
km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=1000000, value=50000)
mileage = st.number_input("Mileage (km/l)", min_value=0.0, max_value=50.0, value=15.0)
engine = st.number_input("Engine (CC)", min_value=500, max_value=5000, value=1200)
max_power = st.number_input("Max Power (bhp)", min_value=20, max_value=500, value=85)
seats = st.number_input("Seats", min_value=2, max_value=12, value=5)

fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
seller_type = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
transmission_type = st.selectbox("Transmission Type", ["Manual", "Automatic"])

# Button to predict
if st.button("Predict Selling Price"):
    # Create DataFrame for input
    input_df = pd.DataFrame({
        'vehicle_age': [vehicle_age],
        'km_driven': [km_driven],
        'mileage': [mileage],
        'engine': [engine],
        'max_power': [max_power],
        'seats': [seats],
        'fuel_type': [fuel_type],
        'seller_type': [seller_type],
        'transmission_type': [transmission_type]
    })

    # === Preprocess input ===
    num_features = ['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
    cat_features = ['seller_type', 'fuel_type', 'transmission_type']

    X_num = scaler.transform(input_df[num_features])
    X_cat = encoder.transform(input_df[cat_features])
    X_processed = np.concatenate([X_num, X_cat], axis=1)

    # Ensure feature order matches training
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
    X_processed_ordered = X_processed_df[feature_names].values

    # === Predict ===
    predicted_price = model.predict(X_processed_ordered)[0]

    st.success(f"💰 Predicted Selling Price: ₹{predicted_price:,.2f}")
