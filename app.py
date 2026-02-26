import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Rocket Thrust Predictor", layout="wide")

st.title("🚀 Rocket Landing Thrust Prediction App")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("model_new.pkl")

model = load_model()

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("Input Parameters")

time_input = st.sidebar.number_input("Time (s)", value=200.0)
altitude_input = st.sidebar.number_input("Altitude (m)", value=30000.0)
velocity_input = st.sidebar.number_input("Velocity (m/s)", value=-200.0)
mass_input = st.sidebar.number_input("Mass (kg)", value=22000.0)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Thrust"):

    input_df = pd.DataFrame([[time_input,
                              altitude_input,
                              velocity_input,
                              mass_input]],
                            columns=['Time','Altitude','Velocity','Mass'])

    prediction = model.predict(input_df)[0]

    st.success(f"Predicted Thrust: {prediction:.2f} N")

# -----------------------------
# OPTIONAL VISUALIZATION
# -----------------------------
st.markdown("---")
st.subheader("Model Demo Visualization")

if st.button("Show Sample Simulation Plot"):

    # Generate small demo timeline
    time_vals = np.linspace(0, 500, 200)
    altitude_vals = np.linspace(80000, 0, 200)
    velocity_vals = np.linspace(-500, 0, 200)
    mass_vals = np.linspace(25000, 20000, 200)

    demo_df = pd.DataFrame({
        'Time': time_vals,
        'Altitude': altitude_vals,
        'Velocity': velocity_vals,
        'Mass': mass_vals
    })

    thrust_preds = model.predict(demo_df)

    fig = plt.figure()
    plt.plot(time_vals, thrust_preds)
    plt.xlabel("Time (s)")
    plt.ylabel("Predicted Thrust (N)")
    plt.title("Predicted Thrust vs Time")

    st.pyplot(fig)

st.markdown("---")
st.info("Model: Random Forest Regressor | Offline Deployment Ready")