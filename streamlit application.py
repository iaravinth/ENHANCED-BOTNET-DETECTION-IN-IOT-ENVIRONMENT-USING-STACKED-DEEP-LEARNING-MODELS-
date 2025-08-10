import base64
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ------------------------------
# Background Image Functions
# ------------------------------
def get_base64(bin_file):
    try:
        with open(bin_file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        st.warning("⚠️ Background image not found. Using default background.")
        return None

def set_background(png_file):
    bin_str = get_base64(png_file)
    if bin_str:
        page_bg_img = f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bin_str}");
            background-size: cover;
        }}
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)

# Set background image (optional)
set_background("1.jpg")

# Load saved model and preprocessing tools
model = load_model('botnet_model.h5')
label_encoder_proto = joblib.load('label_encoder_proto.pkl')
label_encoder_flgs = joblib.load('label_encoder_flgs.pkl')
scaler = joblib.load('scaler.pkl')

# Get the mapping of original labels to encoded values
proto_mapping = dict(zip(label_encoder_proto.classes_, label_encoder_proto.transform(label_encoder_proto.classes_)))
flgs_mapping = dict(zip(label_encoder_flgs.classes_, label_encoder_flgs.transform(label_encoder_flgs.classes_)))

# Title
title = '<h3 style="font-family:Segoe UI Black; color:blue; font-size: 32px;"> IoT Botnet Attack Detection using Hybrid Deep Learning Model (Custom DNN-BiLSTM) </h3>'
st.markdown(title, unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the uploaded CSV
        df = pd.read_csv(uploaded_file)
        st.write("📊 Uploaded Data Preview:")
        st.dataframe(df.head())

        # Drop irrelevant or unnecessary columns (must match training preprocessing)
        drop_cols = ['pkSeqID', 'stime', 'ltime', 'seq', 'smac', 'dmac', 'soui', 'doui', 'sco', 'dco',
                     'spkts', 'dpkts', 'sbytes', 'dbytes', 'srate', 'drate', 'attack', 
                     'saddr', 'daddr', 'sport', 'dport']
        df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

        # Handle missing values
        if 'sport' in df.columns:
            df = df.dropna(subset=['sport'])

        if 'state' in df.columns:
            df = df.drop(columns=['state'])

        # Encoding
        
        # Label encode 'proto' and 'flgs'
        df['proto'] = label_encoder_proto.transform(df['proto'])
        df['flgs'] = label_encoder_flgs.transform(df['flgs'])
        
        # Normalize features
        print(df)
        X_scaled = scaler.transform(df)

        # Reshape for BiLSTM
        X_input = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

        # Prediction button
        if st.button('🔍 Predict'):
            predictions = model.predict([X_input, X_input])
            print(predictions)
            binary_preds = (predictions > 0.5).astype(int).flatten()
            label_map = {0: 'Normal Traffic', 1: 'Botnet Attack'}
            results = [label_map[pred] for pred in binary_preds]  # Map predictions to labels

            # Ensure results exist before accessing
            if len(results) > 0:
                st.markdown('<h3 style="color:white;">🔍 Predicted Result using Custom DNN-BiLSTM:</h3>', unsafe_allow_html=True)
                st.markdown(f'<h3 style="color:darkblue;">{results[0]}</h3>', unsafe_allow_html=True)
            else:
                st.error("⚠️ No predictions were made. Check input data.")

            # Download option
            df['Prediction'] = results  # Store predictions in DataFrame
            csv_output = df.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Download Results as CSV", data=csv_output, file_name="botnet_predictions.csv", mime='text/csv')

    except Exception as e:
        st.error(f"⚠️ An error occurred during processing: {e}")

# Sidebar About Section
st.sidebar.header("📌 About the Project")
st.sidebar.write("""
This project detects **IoT Botnet Attacks** using a **Hybrid Deep Learning Model (DNN-BiLSTM)**. 
It processes network traffic data, encodes categorical features, normalizes data, and predicts whether the traffic is normal or part of a botnet attack.
""")
