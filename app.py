import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- PAGE CONFIG ---
st.set_page_config(page_title="Expense AI Predictor", page_icon="💰", layout="wide")

# --- CUSTOM STYLING (The "Beautiful" Part) ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        background-image: linear-gradient(to right, #6a11cb 0%, #2575fc 100%);
        color: white;
        border: none;
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
    }
    .prediction-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 30px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    with open('expenses_model_final.pkl', 'rb') as f:
        return pickle.load(f)

pkg = load_model()
model = pkg['model']
num_imputer = pkg['num_imputer']
cat_imputer = pkg['cat_imputer']
label_encoders = pkg['label_encoders']
scaler = pkg['scaler']
num_cols = pkg['numeric_features']
cat_cols = pkg['categorical_features']

# --- UI LAYOUT ---
st.title("💰 Expense AI Predictor")
st.markdown("##### Fill in the details below. Missing values are handled by AI.")

# Creating glass-like input containers
with st.container():
    col1, col2, col3 = st.columns([1, 1, 1], gap="medium")
    
    with col1:
        st.subheader("Academic & IQ")
        cgpa = st.text_input("CGPA", placeholder="0.0 - 10.0")
        iq = st.text_input("IQ Score", placeholder="e.g. 100")

    with col2:
        st.subheader("Professional")
        exp = st.text_input("Years of Experience", placeholder="e.g. 5")
        salary = st.text_input("Current Salary", placeholder="e.g. 50000")

    with col3:
        st.subheader("Personal")
        dep = st.text_input("No. of Dependents", placeholder="e.g. 2")
        gender = st.selectbox("Gender", options=["", "Male", "Female"])
        marital = st.selectbox("Marital Status", options=["", "Single", "Married"])

st.write("---")

# --- PREDICTION LOGIC ---
if st.button("CALCULATE ESTIMATED EXPENSES"):
    # 1. Prepare Data
    raw_data = {
        'CGPA': [cgpa if cgpa else np.nan],
        'IQ': [iq if iq else np.nan],
        'Year_of_Experience': [exp if exp else np.nan],
        'Dependents': [dep if dep else np.nan],
        'Salary': [salary if salary else np.nan],
        'Gender': [gender if gender != "" else np.nan],
        'Marital_Status': [marital if marital != "" else np.nan]
    }
    input_df = pd.DataFrame(raw_data)

    try:
        # 2. Impute & Encode
        input_num = num_imputer.transform(input_df[num_cols].astype(float))
        input_cat = cat_imputer.transform(input_df[cat_cols])
        
        input_cat_encoded = np.zeros(input_cat.shape)
        for i, col in enumerate(cat_cols):
            input_cat_encoded[:, i] = label_encoders[col].transform(input_cat[:, i])

        # 3. Scale & Predict
        final_features = np.hstack([input_num, input_cat_encoded])
        final_features_scaled = scaler.transform(final_features)
        prediction = model.predict(final_features_scaled)[0]

        # 4. Beautiful Result Display
        st.markdown(f"""
            <div class="prediction-card">
                <h2 style='color: #2575fc;'>Estimated Monthly Expense</h2>
                <h1 style='font-size: 50px;'>${prediction:,.2f}</h1>
                <p style='color: gray;'>Prediction based on MLR Model with Standard Scaling</p>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Please check your inputs. Error: {e}")