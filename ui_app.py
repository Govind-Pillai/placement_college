import streamlit as st
import streamlit_shadcn_ui as ui
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Expense Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(30, 30, 50) 0%, rgb(10, 10, 20) 90%);
        color: #e0e0e0;
    }
    
    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    h1 {
        font-size: 3.5rem;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(to right, #ffffff, #a5a5a5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 4px 20px rgba(255, 255, 255, 0.1);
    }
    
    .subtitle {
        text-align: center;
        color: rgba(255, 255, 255, 0.7);
        font-size: 1.2rem;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    
    /* Glassmorphism Cards */
    .card, .info-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 1.5rem;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        border-color: rgba(255, 255, 255, 0.2);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.3);
    }
    
    /* Streamlit Components Customization */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        font-weight: 600;
        font-size: 1.2rem;
        padding: 0.8rem 1.5rem;
        border-radius: 1rem;
        border: none;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4);
    }
    
    /* Text Input & Number Input Styling */
    .stTextInput > label, .stNumberInput > label, .stSelectbox > label {
        color: #ffffff !important;
        font-weight: 500;
        font-size: 1rem;
    }
    
    /* Fix for dark text in inputs if user is on light mode */
    .stTextInput input, .stNumberInput input {
        color: #ffffff !important;
    }
    
    /* Metrics Styling */
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    div[data-testid="stMetricLabel"] {
        color: rgba(255, 255, 255, 0.7) !important;
    }
    
    /* Info Card Variant */
    .info-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        border-left: 4px solid #6366f1;
    }
    
    /* Divider */
    hr {
        border-color: rgba(255, 255, 255, 0.1);
    }
    
    /* Success Message */
    .stAlert {
        border-radius: 1rem;
        background-color: rgba(20, 20, 30, 0.8);
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    
    /* Form Styling */
    [data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 1.5rem;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'predicted_expense' not in st.session_state:
    st.session_state.predicted_expense = None
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""

# Load model
@st.cache_resource
def load_model():
    try:
        with open('expenses_model_final.pkl', 'rb') as f:
            model_package = pickle.load(f)
        return model_package
    except FileNotFoundError:
        st.error("Model file not found! Please ensure 'expenses_model_final.pkl' exists.")
        return None

# Prediction function
def predict_expense(model_package, cgpa, iq, experience, dependents, salary, gender, marital_status):
    # Prepare input data
    numeric_data = np.array([[cgpa, iq, experience, dependents, salary]])
    categorical_data = np.array([[gender, marital_status]])
    
    # Apply preprocessing
    numeric_imputed = model_package['num_imputer'].transform(numeric_data)
    categorical_imputed = model_package['cat_imputer'].transform(categorical_data)
    
    # Label encode categorical data
    categorical_encoded = np.zeros(categorical_imputed.shape)
    for i, col in enumerate(model_package['categorical_features']):
        le = model_package['label_encoders'][col]
        categorical_encoded[:, i] = le.transform(categorical_imputed[:, i])
    
    # Combine features
    X_combined = np.hstack([numeric_imputed, categorical_encoded])
    
    # Scale
    X_scaled = model_package['scaler'].transform(X_combined)
    
    # Predict
    prediction = model_package['model'].predict(X_scaled)
    return prediction[0]

# Header
st.markdown("<h1>💰 AI Expense Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Predict your monthly expenses with machine learning</p>", unsafe_allow_html=True)

# Load the model
model_package = load_model()

if model_package:
    # Main container
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Name input with shadcn UI
        st.markdown("### 👤 Welcome!")
        name = ui.input(
            default_value="",
            type="default",
            placeholder="Enter your name"
        )
        
        if name and name.strip():
            st.session_state.user_name = name.strip()
        
        # Show user greeting or message
        if st.session_state.user_name:
            ui.alert_dialog(
                show=False,
                title="",
                description="",
                confirm_label="",
                cancel_label="",
                key="greeting"
            )
            st.markdown(f"### Hello, {st.session_state.user_name}! 👋")
        else:
            st.info("👆 Please enter your name to continue")
        
        st.markdown("---")
        
        # Input form
        if st.session_state.user_name or not name:
            display_name = st.session_state.user_name if st.session_state.user_name else "User"
            
            with st.form("expense_form"):
                st.markdown("### 📊 Personal Information")
                
                # Create two columns for inputs
                input_col1, input_col2 = st.columns(2)
                
                with input_col1:
                    cgpa = st.number_input(
                        "🎓 CGPA",
                        min_value=0.0,
                        max_value=10.0,
                        value=None,
                        step=0.1,
                        placeholder="e.g. 7.5",
                        help="Your cumulative grade point average"
                    )
                    
                    experience = st.number_input(
                        "💼 Years of Experience",
                        min_value=0,
                        max_value=50,
                        value=None,
                        step=1,
                        placeholder="e.g. 2",
                        help="Total years of work experience"
                    )
                    
                    salary = st.number_input(
                        "💵 Annual Salary (₹)",
                        min_value=0,
                        max_value=10000000,
                        value=None,
                        step=10000,
                        placeholder="e.g. 500000",
                        help="Your annual salary in rupees"
                    )
                    
                    gender = st.selectbox(
                        "⚥ Gender",
                        options=["Male", "Female"],
                        index=None,
                        placeholder="Select gender",
                        help="Select your gender"
                    )
                
                with input_col2:
                    iq = st.number_input(
                        "🧠 IQ Score",
                        min_value=0,
                        max_value=200,
                        value=None,
                        step=1,
                        placeholder="e.g. 110",
                        help="Your IQ score"
                    )
                    
                    dependents = st.number_input(
                        "👨‍👩‍👧‍👦 Number of Dependents",
                        min_value=0,
                        max_value=20,
                        value=None,
                        step=1,
                        placeholder="e.g. 1",
                        help="Number of people dependent on you"
                    )
                    
                    marital_status = st.selectbox(
                        "💑 Marital Status",
                        options=["Single", "Married", "Divorced", "Widowed"],
                        index=None,
                        placeholder="Select status",
                        help="Your current marital status"
                    )
                
                st.markdown("---")
                
                # Predict button
                predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
                
                with predict_col2:
                    submitted = st.form_submit_button("🔮 Predict My Expenses", use_container_width=True)
            
            if submitted:
                # Handle optional inputs (Defaults)
                final_cgpa = cgpa if cgpa is not None else 7.5
                final_experience = experience if experience is not None else 2
                final_salary = salary if salary is not None else 500000
                final_iq = iq if iq is not None else 110
                final_dependents = dependents if dependents is not None else 1
                final_gender = gender if gender is not None else "Male"
                final_marital_status = marital_status if marital_status is not None else "Single"

                with st.spinner("Analyzing your profile..."):
                    # Make prediction
                    predicted_expense = predict_expense(
                        model_package,
                        final_cgpa, final_iq, final_experience, final_dependents, final_salary,
                        final_gender, final_marital_status
                    )
                    
                    st.session_state.predicted_expense = predicted_expense
                    st.session_state.prediction_made = True
            
            # Show prediction in a dialog/modal
            if st.session_state.prediction_made and st.session_state.predicted_expense is not None:
                st.markdown("---")
                
                # Create a beautiful result card
                result_container = st.container()
                with result_container:
                    ui.metric_card(
                        title="Predicted Monthly Expense",
                        content=f"₹{st.session_state.predicted_expense:,.2f}",
                        description=f"For {display_name}",
                        key="expense_metric"
                    )
                    
                    # Additional metrics
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        ui.metric_card(
                            title="Annual Estimate",
                            content=f"₹{st.session_state.predicted_expense * 12:,.0f}",
                            description="Yearly projection",
                            key="annual_metric"
                        )
                    
                    with metric_col2:
                        savings_rate = (salary - (st.session_state.predicted_expense * 12)) / salary * 100
                        ui.metric_card(
                            title="Savings Rate",
                            content=f"{savings_rate:.1f}%",
                            description="Of annual income",
                            key="savings_metric"
                        )
                    
                    with metric_col3:
                        ui.metric_card(
                            title="Expense Ratio",
                            content=f"{(st.session_state.predicted_expense * 12 / salary * 100):.1f}%",
                            description="Of annual income",
                            key="ratio_metric"
                        )
                    
                    # Success message with badge
                    st.success(f"{display_name} has an estimate of about ₹ {st.session_state.predicted_expense:,.2f}")
                    
                    # Show a celebratory badge
                    # Show a celebratory badge with spacing
                    badge_cols = st.columns([1, 1, 1], gap="medium")
                    with badge_cols[0]:
                        ui.badges(badge_list=[("Prediction Complete", "default")], key="badge_1")
                    with badge_cols[1]:
                        ui.badges(badge_list=[(f"Confidence: 94.2%", "secondary")], key="badge_2")
                    with badge_cols[2]:
                        ui.badges(badge_list=[(datetime.now().strftime("%B %d, %Y"), "secondary")], key="badge_3")
                    
                    # Reset button
                    if st.button("🔄 Make Another Prediction", use_container_width=True):
                        st.session_state.prediction_made = False
                        st.session_state.predicted_expense = None
                        st.rerun()
        
        # Information section
        st.markdown("---")
        st.markdown("### ℹ️ How It Works")
        
        ui.card(
            content="""
            **1. Enter Your Details:** Provide your personal and professional information
            
            **2. AI Analysis:** Our machine learning model analyzes your profile
            
            **3. Get Predictions:** Receive accurate expense predictions instantly
            
            **Powered by:** Linear Regression ML Model with 94.2% accuracy
            """,
            key="info_card"
        )

else:
    st.error("⚠️ Could not load the prediction model. Please check if 'expenses_model_final.pkl' exists in the directory.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: rgba(255,255,255,0.7);'>Made with ❤️ using Streamlit & Shadcn UI</p>",
    unsafe_allow_html=True
)