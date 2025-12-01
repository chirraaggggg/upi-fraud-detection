import streamlit as st
import pandas as pd
import datetime
import requests
import json
from datetime import datetime as dt

"""
# Welcome to your own UPI Transaction Fraud Detector!

You have the option of inspecting a single transaction by adjusting the parameters below OR you can even check 
multiple transactions at once by uploading a .csv file in the specified format
"""

# API endpoint (update this if your FastAPI server is running on a different URL)
API_URL = "http://127.0.0.1:8000"

# Transaction type options
tt = ["Bill Payment", "Investment", "Other", "Purchase", "Refund", "Subscription"]
pg = ["Google Pay", "HDFC", "ICICI UPI", "IDFC UPI", "Other", "Paytm", "PhonePe", "Razor Pay"]
ts = ['Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal']
mc = ['Donations and Devotion', 'Financial services and Taxes', 'Home delivery', 'Investment', 'More Services', 'Other', 'Purchases', 'Travel bookings', 'Utilities']

# Sidebar for API status
st.sidebar.title("API Status")
try:
    response = requests.get(f"{API_URL}/")
    if response.status_code == 200:
        st.sidebar.success("✅ API is running")
    else:
        st.sidebar.warning("⚠️ API returned an unexpected response")
except requests.exceptions.RequestException as e:
    st.sidebar.error(f"❌ Could not connect to API: {e}")
    st.sidebar.info("Make sure to start the FastAPI server first by running: 'uvicorn app:app --reload'")

# Main app
st.title("🔍 UPI Transaction Fraud Detection")
st.write("Enter transaction details to check for potential fraud")

# Input form
with st.form("transaction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        tran_date = st.date_input("Transaction Date", datetime.date.today())
        tran_type = st.selectbox("Transaction Type", tt)
        pmt_gateway = st.selectbox("Payment Gateway", pg)
    
    with col2:
        amt = st.number_input("Amount (₹)", min_value=0.01, step=0.01, format="%.2f")
        tran_state = st.selectbox("Transaction State", ts)
        merch_cat = st.selectbox("Merchant Category", mc)
    
    submitted = st.form_submit_button("Check for Fraud")

# Process form submission
if submitted:
    try:
        # Prepare the request data
        transaction_data = {
            "amount": float(amt),
            "date": tran_date.strftime("%Y-%m-%d"),
            "transaction_type": tran_type,
            "payment_gateway": pmt_gateway,
            "transaction_state": tran_state,
            "merchant_category": merch_cat
        }
        
        # Make API request
        with st.spinner("Analyzing transaction..."):
            response = requests.post(f"{API_URL}/predict", json=transaction_data)
            
            if response.status_code == 200:
                result = response.json()
                
                # Display results
                st.subheader("Analysis Results")
                
                if result["is_fraud"]:
                    st.error(f"🚨 High Risk of Fraud Detected! ({result['fraud_probability']*100:.2f}%)")
                    st.warning("This transaction appears to be suspicious. Please verify the details and proceed with caution.")
                else:
                    st.success(f"✅ Transaction appears to be legitimate ({result['fraud_probability']*100:.2f}% risk)")
                    st.info("No significant fraud indicators detected.")
                
                # Show transaction details
                with st.expander("View Transaction Details"):
                    st.json(transaction_data)
                
                # Show probability gauge
                st.subheader("Fraud Probability")
                st.progress(result["fraud_probability"])
                
            else:
                st.error(f"Error: {response.text}")
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Add some information about the app
st.markdown("---")
st.markdown("""
### About this App
This application uses machine learning to detect potentially fraudulent UPI transactions. 
The model analyzes various transaction patterns and characteristics to assess the risk of fraud.

**Note:** This is a prediction based on machine learning and may not be 100% accurate. 
Always verify suspicious transactions
""")
