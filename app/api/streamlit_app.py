# app/api/streamlit_app.py
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import streamlit as st
from app.pipeline.main import inference

# Set up Streamlit UI
st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detector")
st.write("Enter a news **title** and **content** to detect if it's *Fake* or *True*.")

# Input fields
title = st.text_input("📝 News Title", placeholder="e.g. Breaking: New discovery in space exploration")
text = st.text_area("📰 News Text", placeholder="Paste the news content here...", height=200)

# Optional model selection
model = st.selectbox("🔍 Choose a model", ["logistic", "naive_bayes", "random_forest"], index=0)

# Prediction trigger
if st.button("🔎 Analyze"):
    if not title.strip() or not text.strip():
        st.warning("⚠️ Please enter both a title and the news text.")
    else:
        with st.spinner("Analyzing..."):
            prediction = inference(title, text, model_type=model)
            st.success(f"✅ Prediction: **{prediction} News**")
