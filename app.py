from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

app = FastAPI(title="UPI Fraud Detection API",
             description="API for detecting fraudulent UPI transactions",
             version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model and encoders
model = joblib.load("UPI Fraud Detection Final.pkl")

# Define the input data model
class Transaction(BaseModel):
    amount: float
    date: str  # Format: "YYYY-MM-DD"
    transaction_type: str
    payment_gateway: str
    transaction_state: str
    merchant_category: str

# Categories for one-hot encoding
transaction_types = ["Bill Payment", "Investment", "Other", "Purchase", "Refund", "Subscription"]
payment_gateways = ["Google Pay", "HDFC", "ICICI UPI", "IDFC UPI", "Other", "Paytm", "PhonePe", "Razor Pay"]
transaction_states = ['Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal']
merchant_categories = ['Donations and Devotion', 'Financial services and Taxes', 'Home delivery', 'Investment', 'More Services', 'Other', 'Purchases', 'Travel bookings', 'Utilities']

@app.get("/")
async def root():
    return {"message": "Welcome to the UPI Fraud Detection API. Use /predict endpoint to make predictions."}

@app.post("/predict")
async def predict(transaction: Transaction):
    try:
        # Parse date
        trans_date = datetime.strptime(transaction.date, "%Y-%m-%d")
        month = trans_date.month
        year = trans_date.year
        
        # Create one-hot encoded vectors
        tt_oh = [1 if t == transaction.transaction_type else 0 for t in transaction_types]
        pg_oh = [1 if p == transaction.payment_gateway else 0 for p in payment_gateways]
        ts_oh = [1 if s == transaction.transaction_state else 0 for s in transaction_states]
        mc_oh = [1 if m == transaction.merchant_category else 0 for m in merchant_categories]
        
        # Prepare feature vector
        features = [transaction.amount, year, month] + tt_oh + pg_oh + ts_oh + mc_oh
        
        # Make prediction
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0][1]  # Probability of being fraudulent
        
        return {
            "is_fraud": bool(prediction),
            "fraud_probability": float(probability),
            "transaction_details": {
                "amount": transaction.amount,
                "date": transaction.date,
                "transaction_type": transaction.transaction_type,
                "payment_gateway": transaction.payment_gateway,
                "transaction_state": transaction.transaction_state,
                "merchant_category": transaction.merchant_category
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
