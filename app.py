import os
import joblib
import streamlit as st

@st.cache_resource
def load_model():
    if not os.path.exists("model_new.pkl"):
        st.error("❌ model_new.pkl NOT found. Upload it to repo.")
        st.stop()
    return joblib.load("model_new.pkl")

model = load_model()
