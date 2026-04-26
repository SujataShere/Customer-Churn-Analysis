import streamlit as st
import pandas as pd
import pickle

# Load model and columns
model = pickle.load(open("model.pkl", "rb"))
cols = pickle.load(open("columns.pkl", "rb"))

# Page setup
st.set_page_config(page_title="Customer Churn App", page_icon="📊")

# Title
st.title("📊 Customer Churn Prediction App")
st.write("Enter customer details to predict churn")

# Sidebar inputs (better UI)
st.sidebar.header("Customer Inputs")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 50.0)
total_charges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 500.0)

contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

payment = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

# Prediction button
if st.sidebar.button("Predict"):

    # Create empty dataframe with same columns as training
    input_df = pd.DataFrame(columns=cols)
    input_df.loc[0] = 0

    # Fill numeric values
    if "tenure" in input_df.columns:
        input_df["tenure"] = tenure

    if "MonthlyCharges" in input_df.columns:
        input_df["MonthlyCharges"] = monthly_charges

    if "TotalCharges" in input_df.columns:
        input_df["TotalCharges"] = total_charges

    # Encode Contract
    contract_col = f"Contract_{contract}"
    if contract_col in input_df.columns:
        input_df[contract_col] = 1

    # Encode Payment Method
    payment_col = f"PaymentMethod_{payment}"
    if payment_col in input_df.columns:
        input_df[payment_col] = 1

    # Prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # Output
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to CHURN\n\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Customer is NOT likely to churn\n\nProbability: {probability:.2f}")

    # Show input data (for debugging / demo)
    st.write("### Input Data Sent to Model")
    st.dataframe(input_df)