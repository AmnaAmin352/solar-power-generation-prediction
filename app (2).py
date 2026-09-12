import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(page_title="Solar Power Predictor", page_icon="☀️", layout="centered")
st.title("☀️ Solar Power Generation Predictor")
st.write("Enter weather and plant details below to predict AC Power output (kW).")

# ------------------------------------------------------------
# Load trained model + scaler
# (Make sure gbr_solar_model.pkl and scaler.pkl are in the SAME folder as this app.py)
# ------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("gbr_solar_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.error(
        "Model files not found. Please put 'gbr_solar_model.pkl' and 'scaler.pkl' "
        "in the same folder as this app.py, then restart the app."
    )

# ------------------------------------------------------------
# Input form
# ------------------------------------------------------------
with st.form("prediction_form"):
    st.subheader("Input Values")

    col1, col2 = st.columns(2)

    with col1:
        ambient_temp = st.number_input(
            "Ambient Temperature (°C)", min_value=0.0, max_value=60.0, value=32.0, step=0.5
        )
        module_temp = st.number_input(
            "Module Temperature (°C)", min_value=0.0, max_value=80.0, value=45.0, step=0.5
        )
        irradiation = st.number_input(
            "Irradiation (0 to 1.2)", min_value=0.0, max_value=1.2, value=0.75, step=0.01
        )

    with col2:
        hour = st.slider("Hour of Day (0-23)", min_value=0, max_value=23, value=13)
        plant_id = st.selectbox("Plant", options=[1, 2], format_func=lambda x: f"Plant {x}")

    submitted = st.form_submit_button("Predict Power Output")

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------
if submitted:
    if not model_loaded:
        st.warning("Cannot predict — model files are missing.")
    else:
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        # IMPORTANT: column names and order must exactly match training feature_cols
        new_data = pd.DataFrame([{
            "AMBIENT_TEMPERATURE": ambient_temp,
            "MODULE_TEMPERATURE": module_temp,
            "IRRADIATION": irradiation,
            "HOUR_SIN": hour_sin,
            "HOUR_COS": hour_cos,
            "PLANT_ID": plant_id,
        }])

        new_data_scaled = scaler.transform(new_data)
        prediction = model.predict(new_data_scaled)[0]
        prediction = max(prediction, 0)  # power output can't be negative

        st.success(f"### Predicted AC Power Output: **{prediction:.2f} kW**")

        with st.expander("See input values used"):
            st.dataframe(new_data)

st.markdown("---")
st.caption(
    "Model: Gradient Boosting Regressor | Trained on Kaggle 'Solar Power Generation Data' "
    "(Plant 1 + Plant 2, combined)."
)