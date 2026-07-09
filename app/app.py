import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and feature names
model = joblib.load('/Users/darshanv/credit-risk-scorer/models/champion_lightgbm_v2.pkl')
feature_names = joblib.load('/Users/darshanv/credit-risk-scorer/models/feature_names_v2.pkl')

st.set_page_config(page_title="Credit Risk Scorer", page_icon="🏦", layout="centered")

st.title("🏦 Credit Risk Scorer")
st.markdown("Predict loan default probability for thin-file borrowers using behavioral and financial signals.")
st.divider()

st.subheader("Applicant Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (years)", 20, 70, 35)
    income = st.number_input("Annual Income (₹)", 50000, 10000000, 200000, step=10000)
    loan_amount = st.number_input("Loan Amount (₹)", 50000, 5000000, 500000, step=10000)
    annuity = st.number_input("Monthly Repayment (₹)", 1000, 100000, 15000, step=500)
    employment_years = st.slider("Years at current job", 0, 40, 5)

with col2:
    ext_source_2 = st.slider("External Credit Score 2", 0.0, 1.0, 0.5, 0.01)
    ext_source_3 = st.slider("External Credit Score 3", 0.0, 1.0, 0.5, 0.01)
    ext_source_1 = st.slider("External Credit Score 1", 0.0, 1.0, 0.5, 0.01)
    instal_late_ratio = st.slider("Fraction of late installment payments", 0.0, 1.0, 0.0, 0.01)
    bureau_debt_ratio = st.slider("Bureau debt-to-credit ratio", 0.0, 2.0, 0.3, 0.01)

st.divider()

if st.button("Calculate Default Risk", type="primary"):

    # Build input row with all features set to median/zero
    input_data = pd.DataFrame(
        np.zeros((1, len(feature_names))),
        columns=feature_names
    )

    # Fill in the values the user provided
    days_birth = -age * 365
    days_employed = -employment_years * 365

    feature_map = {
        'EXT_SOURCE_1': ext_source_1,
        'EXT_SOURCE_2': ext_source_2,
        'EXT_SOURCE_3': ext_source_3,
        'AMT_INCOME_TOTAL': income,
        'AMT_CREDIT': loan_amount,
        'AMT_ANNUITY': annuity,
        'DAYS_BIRTH': days_birth,
        'DAYS_EMPLOYED': days_employed,
        'age_years': age,
        'income_annuity_ratio': income / annuity if annuity > 0 else 0,
        'employment_to_age_ratio': employment_years / age if age > 0 else 0,
        'credit_to_income_ratio': loan_amount / income if income > 0 else 0,
        'instal_late_payment_ratio': instal_late_ratio,
        'bureau_debt_to_credit_ratio': bureau_debt_ratio,
    }

    for feature, value in feature_map.items():
        if feature in input_data.columns:
            input_data[feature] = value

    # Predict
    proba = model.predict_proba(input_data)[0][1]

    # Display result
    st.subheader("Risk Assessment")

    if proba < 0.2:
        risk_tier = "🟢 Low Risk"
        color = "green"
        recommendation = "Recommend approval. Strong repayment likelihood."
    elif proba < 0.4:
        risk_tier = "🟡 Medium Risk"
        color = "orange"
        recommendation = "Consider approval with standard due diligence."
    else:
        risk_tier = "🔴 High Risk"
        color = "red"
        recommendation = "Flag for manual review. Elevated default probability."

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Default Probability", f"{proba:.1%}")
    with col2:
        st.metric("Risk Tier", risk_tier)

    st.markdown(f"**Recommendation:** {recommendation}")

    st.divider()
    st.caption("Model: LightGBM V2 | ROC-AUC: 0.7755 | Trained on Home Credit Default Risk Dataset")
    st.caption("⚠️ This tool is for demonstration purposes only and should not be used as the sole basis for lending decisions.")