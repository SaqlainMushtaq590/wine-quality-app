import streamlit as st
import pandas as pd
import numpy as np
import pickle # Used to load the .pkl files
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Wine Quality AI", layout="wide")
st.title("🍷 Wine Quality Prediction System")
st.markdown("Enter the chemical details below to see if the AI classifies the wine as **Good** or **Bad**.")


import os

# Ye line add karo file ke top pe
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# --- 2. LOAD THE SAVED BRAINS (PICKLE) ---
@st.cache_resource # Resource is used for external files like models
def load_models():
    # Loading the files we saved in Jupyter
    with open('wine_Scaler.pkl', 'rb') as f:
        sc = pickle.load(f)
    with open('Log_Model.pkl', 'rb') as f:
        log = pickle.load(f)
    with open('SVM_Model.pkl', 'rb') as f:
        svm = pickle.load(f)
    with open('DTC.pkl', 'rb') as f:
        dt = pickle.load(f)
    return sc, log, svm, dt

# Run the loader
try:
    scaler, log_model, svm_model, dt_model = load_models()
except FileNotFoundError:
    st.error("⚠️ Model files not found! Make sure the .pkl files are in the same folder as this app.")

# --- 3. KEYBOARD INPUT FORM ---
st.header("⌨️ Lab Results Input")
col_a, col_b = st.columns(2)

user_data = {}

with col_a:
    user_data['fixed acidity'] = st.number_input('Fixed Acidity', value=7.4, format="%.2f")
    user_data['volatile acidity'] = st.number_input('Volatile Acidity (Vinegar)', value=0.70, format="%.2f")
    user_data['citric acid'] = st.number_input('Citric Acid', value=0.00, format="%.2f")
    user_data['residual sugar'] = st.number_input('Residual Sugar', value=1.9, format="%.2f")
    user_data['chlorides'] = st.number_input('Chlorides', value=0.076, format="%.3f")
    user_data['free sulfur dioxide'] = st.number_input('Free Sulfur Dioxide', value=11.0, format="%.1f")

with col_b:
    user_data['total sulfur dioxide'] = st.number_input('Total Sulfur Dioxide', value=34.0, format="%.1f")
    user_data['density'] = st.number_input('Density', value=0.9978, format="%.4f")
    user_data['pH'] = st.number_input('pH Level', value=3.51, format="%.2f")
    user_data['sulphates'] = st.number_input('Sulphates', value=0.56, format="%.2f")
    user_data['alcohol'] = st.number_input('Alcohol Percentage', value=9.4, format="%.1f")

# --- 4. PREDICTION LOGIC ---
if st.button('🚀 Analyze Wine Quality'):
    # Convert input to DataFrame
    input_df = pd.DataFrame([user_data])
    
    # Scale for math-heavy models (Logistic and SVM)
    input_scaled = scaler.transform(input_df)
    
    # Get Predictions
    prob = log_model.predict_proba(input_scaled)[0][1]
    svm_res = svm_model.predict(input_scaled)[0]
    dt_res = dt_model.predict(input_df)[0] # Decision Tree uses raw data
    
    st.divider()
    
    # Majority Vote (2 out of 3)
    final_score = (prob > 0.5) + svm_res + dt_res
    
    if final_score >= 2:
        st.success("### Prediction: THIS IS A GOOD QUALITY WINE! 🌟")
        st.balloons()
    else:
        st.error("### Prediction: THIS IS AN AVERAGE/BAD WINE. 📉")

    # Display results from all 3 'brains'
    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric("Sigmoid Probability", f"{prob:.2%}")
    res_col2.metric("SVM Boundary", "Good" if svm_res == 1 else "Bad")
    res_col3.metric("Decision Tree", "Good" if dt_res == 1 else "Bad")

# --- 5. DATA VISUALIZATION (Optional) ---
# We still load the CSV here just for the graph background
if st.checkbox('Show Alcohol Comparison Graph'):
    df_raw = pd.read_csv('winequality-red.csv')
    fig, ax = plt.subplots()
    sns.histplot(data=df_raw, x='alcohol', color='gray', alpha=0.3, ax=ax)
    plt.axvline(user_data['alcohol'], color='red', linestyle='--', label='Your Input')
    plt.title("Your Wine vs Global Dataset")
    plt.legend()
    st.pyplot(fig)
