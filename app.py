import streamlit as st
import pandas as pd
import joblib


# ==============================
# Load Model
# ==============================

model = joblib.load("churn_model.pkl")
model_features = joblib.load("model_features.pkl")


# ==============================
# Page Configuration
# ==============================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered"
)


# ==============================
# Header
# ==============================

st.title("📊 Customer Churn Predictor")

st.write(
    "Predict whether a customer is likely to leave the service "
    "based on their account and service information."
)

st.divider()


# ==============================
# Customer Information
# ==============================

st.subheader("👤 Customer Information")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=100,
        value=12
    )

with col2:
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0
)


# ==============================
# Contract & Services
# ==============================

st.subheader("📄 Contract & Services")

col1, col2 = st.columns(2)

with col1:
    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )

with col2:
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

col1, col2 = st.columns(2)

with col1:
    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

with col2:
    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )


# ==============================
# Billing
# ==============================

st.subheader("💳 Billing Information")

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)


st.divider()


# ==============================
# Prediction
# ==============================

if st.button("🔮 Predict Customer Churn", use_container_width=True):

    # Create input dataframe
    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=model_features
    )

    # Numerical values
    if "tenure" in input_data.columns:
        input_data["tenure"] = tenure

    if "MonthlyCharges" in input_data.columns:
        input_data["MonthlyCharges"] = monthly_charges

    if "TotalCharges" in input_data.columns:
        input_data["TotalCharges"] = total_charges


    # Contract
    if contract == "One year":
        if "Contract_One year" in input_data.columns:
            input_data["Contract_One year"] = 1

    elif contract == "Two year":
        if "Contract_Two year" in input_data.columns:
            input_data["Contract_Two year"] = 1


    # Internet Service
    if internet_service == "Fiber optic":
        if "InternetService_Fiber optic" in input_data.columns:
            input_data["InternetService_Fiber optic"] = 1

    elif internet_service == "No":
        if "InternetService_No" in input_data.columns:
            input_data["InternetService_No"] = 1


    # Tech Support
    if tech_support == "Yes":
        if "TechSupport_Yes" in input_data.columns:
            input_data["TechSupport_Yes"] = 1

    elif tech_support == "No internet service":
        if "TechSupport_No internet service" in input_data.columns:
            input_data["TechSupport_No internet service"] = 1


    # Online Security
    if online_security == "Yes":
        if "OnlineSecurity_Yes" in input_data.columns:
            input_data["OnlineSecurity_Yes"] = 1

    elif online_security == "No internet service":
        if "OnlineSecurity_No internet service" in input_data.columns:
            input_data["OnlineSecurity_No internet service"] = 1


    # Payment Method
    if payment_method == "Electronic check":
        if "PaymentMethod_Electronic check" in input_data.columns:
            input_data["PaymentMethod_Electronic check"] = 1

    elif payment_method == "Mailed check":
        if "PaymentMethod_Mailed check" in input_data.columns:
            input_data["PaymentMethod_Mailed check"] = 1

    elif payment_method == "Credit card (automatic)":
        if "PaymentMethod_Credit card (automatic)" in input_data.columns:
            input_data["PaymentMethod_Credit card (automatic)"] = 1


    # ==============================
    # Make Prediction
    # ==============================

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]


    # ==============================
    # Display Result
    # ==============================

    st.divider()

    st.subheader("📈 Prediction Result")

    st.metric(
        "Churn Probability",
        f"{probability:.2%}"
    )

    st.progress(float(probability))


    if prediction == 1:

        st.error(
            "⚠️ Customer is likely to churn."
        )

        st.write(
            "The model identifies this customer as a potential churn customer."
        )

    else:

        st.success(
            "✅ Customer is unlikely to churn."
        )

        st.write(
            "The model identifies this customer as less likely to churn."
        )


    # Risk level
    if probability >= 0.70:

        st.warning("🔴 Higher churn probability")

    elif probability >= 0.40:

        st.warning("🟠 Moderate churn probability")

    else:

        st.info("🟢 Lower churn probability")


# ==============================
# Footer
# ==============================

st.divider()

st.caption(
    "Machine Learning Project | Customer Churn Prediction | "
    "Python • Scikit-learn • Streamlit"
)