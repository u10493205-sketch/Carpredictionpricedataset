import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ----------------------------------------------------------
# 🔧 Define paths for model and preprocessing files
# ----------------------------------------------------------
BASE_DIR = r"C:\Users\E105484\OneDrive - Road Accident Fund\Documents\Regynisis\StreamlitApp"

MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "one_hot_encoder.pkl")
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, "feature_names.pkl")

# ----------------------------------------------------------
# 🧠 Load saved objects
# ----------------------------------------------------------
@st.cache_resource
def load_objects():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        encoder = joblib.load(ENCODER_PATH)
        feature_names = joblib.load(FEATURE_NAMES_PATH)
        st.success("✅ Model and preprocessing files loaded successfully!")
        return model, scaler, encoder, feature_names
    except Exception as e:
        st.error(f"❌ Error loading files: {e}")
        return None, None, None, None

model, scaler, encoder, feature_names = load_objects()

# ----------------------------------------------------------
# 🎨 Streamlit UI
# ----------------------------------------------------------
st.set_page_config(page_title="Car Price Prediction", layout="centered")

st.title("🚗 Car Price Prediction App")
st.markdown("Enter car details below to estimate its **selling price (₹)**.")

# ----------------------------------------------------------
# 📋 Input features
# ----------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    vehicle_age = st.slider("Vehicle Age (years)", 0, 30, 5)
    km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=200000, value=50000)
    mileage = st.number_input("Mileage (km/l)", min_value=5.0, max_value=40.0, value=18.0)
with col2:
    engine = st.number_input("Engine (CC)", min_value=500, max_value=5000, value=1200)
    max_power = st.number_input("Max Power (bhp)", min_value=20.0, max_value=400.0, value=85.0)
    seats = st.number_input("Number of Seats", min_value=2, max_value=10, value=5)

# Categorical options
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual", "Trustmark Dealer"])
transmission_type = st.selectbox("Transmission Type", ["Manual", "Automatic"])

# ----------------------------------------------------------
# 🧮 Preprocessing for prediction
# ----------------------------------------------------------
def preprocess_input():
    input_dict = {
        "vehicle_age": vehicle_age,
        "km_driven": km_driven,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats,
        "fuel_type": fuel_type,
        "seller_type": seller_type,
        "transmission_type": transmission_type
    }

    input_df = pd.DataFrame([input_dict])

    # Numerical and categorical columns
    num_features = ["vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats"]
    cat_features = ["fuel_type", "seller_type", "transmission_type"]

    # Scale numerical features
    scaled_values = scaler.transform(input_df[num_features])

    # Encode categorical features
    encoded_values = encoder.transform(input_df[cat_features]).toarray()

    # Combine numerical and categorical features
    processed = np.concatenate([scaled_values, encoded_values], axis=1)
    processed_df = pd.DataFrame(processed, columns=feature_names)

    return processed_df

# ----------------------------------------------------------
# 🚀 Prediction
# ----------------------------------------------------------
if st.button("🔍 Predict Price"):
    if model is not None:
        try:
            processed_input = preprocess_input()
            prediction = model.predict(processed_input)[0]
            st.success(f"💰 Predicted Selling Price: ₹{prediction:,.2f}")
        except Exception as e:
            st.error(f"⚠️ Prediction error: {e}")
    else:
        st.error("❌ Model not loaded. Please check your files.")

# ----------------------------------------------------------
# ℹ️ Footer
# ----------------------------------------------------------
st.markdown("---")
st.caption("Developed for Regenesys Data Science Project — Car Price Prediction 🚗")


