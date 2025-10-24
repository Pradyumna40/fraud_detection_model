import streamlit as st
import pandas as pd
import joblib
import lightgbm as lgb  # Keep this import, joblib may need it to load the model
from geopy.distance import geodesic

# --- Page Configuration ---
# Set page title, icon, and layout
st.set_page_config(
    page_title="FinSafe Fraud Detector",
    layout="wide",
    page_icon="🛡️"
)

# --- Model Loading ---
# Cache the models so they only load once
@st.cache_resource
def load_models():
    """Loads the model and encoder from disk."""
    try:
        model = joblib.load("fraud_detection_model.jb")
        encoder = joblib.load("label_encoder.jb")
        return model, encoder
    except FileNotFoundError:
        st.error("Model or encoder file not found. Please ensure 'fraud_detection_model.jb' and 'label_encoder.jb' are in the correct path.")
        return None, None

model, encoder = load_models()

# --- Helper Function ---
def haversine(lat1, lon1, lat2, lon2):
    """Calculate the distance between two points on Earth."""
    try:
        return geodesic((lat1, lon1), (lat2, lon2)).km
    except ValueError:
        return 0 # Handle potential invalid lat/lon inputs

# --- UI Layout ---
st.title("🛡️ FinSafe Fraud Detection System")
st.write("Enter the transaction details below to check for potential fraud.")

# Use a form to group all inputs
with st.form(key="transaction_form"):
    
    # --- Group 1: Transaction Details ---
    st.subheader("Transaction Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        merchant = st.text_input("Merchant Name")
    with col2:
        category = st.text_input("Transaction Category")
    with col3:
        amt = st.number_input("Transaction Amount ($)", min_value=0.0, format="%.2f")

    # --- Group 2: Location Details ---
    st.subheader("Location Details")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**User Location**")
        lat = st.number_input("Your Latitude", format="%.6f", value=0.0)
        long = st.number_input("Your Longitude", format="%.6f", value=0.0)
    with col2:
        st.markdown("**Merchant Location**")
        merch_lat = st.number_input("Merchant Latitude", format="%.6f", value=0.0)
        merch_long = st.number_input("Merchant Longitude", format="%.6f", value=0.0)

    # --- Group 3: Time & Customer Details ---
    st.subheader("Time & Customer Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        hour = st.slider("Transaction Hour", 0, 23, 12)
        day = st.slider("Transaction Day", 1, 31, 15)
    with col2:
        month = st.slider("Transaction Month", 1, 12, 6)
        gender = st.selectbox("Gender", ["Male", "Female"], index=0)
    with col3:
        cc_num = st.text_input("Credit Card Number", 
                               help="This will be hashed locally before prediction.")

    st.divider()

    # --- Submit Button ---
    submitted = st.form_submit_button(
        "Check For Fraud",
        use_container_width=True,
        type="primary"
    )

# --- Prediction Logic ---
# This block only runs *after* the "submitted" button is pressed
if submitted:
    # Ensure models are loaded before proceeding
    if not model or not encoder:
        st.stop()

    # Simple validation
    if not all([merchant, category, cc_num]):
        st.error("Please fill all text fields: Merchant, Category, and Credit Card Number.")
    else:
        try:
            # 1. Calculate distance
            distance = haversine(lat, long, merch_lat, merch_long)
            
            # 2. Create DataFrame
            input_data = pd.DataFrame(
                [[merchant, category, amt, distance, hour, day, month, gender, cc_num]],
                columns=['merchant', 'category', 'amt', 'distance', 'hour', 'day', 'month', 'gender', 'cc_num']
            )
            
            # 3. Preprocessing
            categorical_col = ['merchant', 'category', 'gender']
            input_data_processed = input_data.copy() # Avoid modifying the original
            
            for col in categorical_col:
                try:
                    # Transform data using the loaded encoder
                    input_data_processed[col] = encoder[col].transform(input_data_processed[col])
                except ValueError:
                    # Handle unseen labels by setting to a default (e.g., -1)
                    input_data_processed[col] = -1
            
            # 4. Hash Credit Card Number
            input_data_processed['cc_num'] = input_data_processed['cc_num'].apply(lambda x: hash(x) % (10 ** 2))
            
            # 5. Prediction
            prediction = model.predict(input_data_processed)[0]
            
            # 6. Display Result
            if prediction == 1:
                st.error("Prediction: Fraudulent Transaction 🚨", icon="🚨")
            else:
                st.success("Prediction: Legitimate Transaction ✅", icon="✅")

            # Optional: Show the data that was sent to the model
            with st.expander("See processed data sent to model"):
                st.dataframe(input_data_processed)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")