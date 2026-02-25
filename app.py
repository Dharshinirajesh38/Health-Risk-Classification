# ============================================
# Healthcare Risk Classification - Hospital UI
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# --------------------------------------------
# Page Configuration
# --------------------------------------------

st.set_page_config(
    page_title="Healthcare Risk System",
    page_icon="🏥",
    layout="wide"
)

# --------------------------------------------
# Simple Login System
# --------------------------------------------

def login():

    st.title("🏥 Hospital Risk Management System")
    st.markdown("### Secure Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "health123":
            st.session_state["logged_in"] = True
        else:
            st.error("Invalid Username or Password")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
    st.stop()
# --------------------------------------------
# Logout Button (Sidebar)
# --------------------------------------------

st.sidebar.markdown("---")
if st.sidebar.button("🚪 Logout"):
    st.session_state["logged_in"] = False
    st.rerun()
# --------------------------------------------
# Load Model & Metrics
# --------------------------------------------

model = joblib.load("models/risk_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
accuracy = joblib.load("models/test_accuracy.pkl")
cv_scores = joblib.load("models/cv_scores.pkl")
cm = joblib.load("models/confusion_matrix.pkl")
report = joblib.load("models/classification_report.pkl")

# --------------------------------------------
# Hospital Header
# --------------------------------------------

st.markdown("""
<style>
.big-font {
    font-size:32px !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">🏥 Healthcare Risk Classification Dashboard</p>', unsafe_allow_html=True)
st.markdown("---")

# --------------------------------------------
# Sidebar - Performance
# --------------------------------------------

st.sidebar.header("📊 Model Performance")
st.sidebar.success(f"Test Accuracy: {accuracy:.2f}")
st.sidebar.info(f"Mean CV Accuracy: {np.mean(cv_scores):.2f}")

# --------------------------------------------
# Layout Columns
# --------------------------------------------

col1, col2 = st.columns(2)

# --------------------------------------------
# Confusion Matrix
# --------------------------------------------

with col1:
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

# --------------------------------------------
# Feature Importance
# --------------------------------------------

with col2:
    st.subheader("Top Contributing Features")

    classifier = model.named_steps["classifier"]
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    coefficients = classifier.coef_[0]

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients
    })

    importance_df["Absolute"] = importance_df["Coefficient"].abs()
    importance_df = importance_df.sort_values("Absolute", ascending=False)

    st.bar_chart(
        importance_df.set_index("Feature")["Absolute"].head(10)
    )

st.markdown("---")

# --------------------------------------------
# Patient Input Section
# --------------------------------------------

st.subheader("🩺 Patient Risk Assessment")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 25, 85, 40)
    bmi = st.number_input("BMI", 15.0, 45.0, 25.0)
    cholesterol = st.number_input("Cholesterol", 100, 350, 200)
    hba1c = st.number_input("HbA1c", 4.0, 15.0, 6.0)

with col2:
    systolic_bp = st.number_input("Systolic BP", 90, 200, 120)
    diastolic_bp = st.number_input("Diastolic BP", 60, 120, 80)
    heart_rate = st.number_input("Heart Rate", 50, 120, 70)
    steps = st.number_input("Daily Steps", 0, 20000, 5000)

with col3:
    gender = st.selectbox("Gender", ["Male", "Female"])
    smoking = st.selectbox("Smoking Status",
                           ["Non-Smoker", "Former Smoker", "Current Smoker"])
    alcohol = st.slider("Alcohol/week", 0, 20, 2)
    activity = st.slider("Activity hours/week", 0.0, 15.0, 3.0)
    sleep = st.slider("Sleep hours", 3.0, 10.0, 7.0)
    family_diabetes = st.selectbox("Family History Diabetes", [0, 1])
    family_heart = st.selectbox("Family History Heart Disease", [0, 1])

# --------------------------------------------
# Prediction Button
# --------------------------------------------

if st.button("🔍 Evaluate Risk"):

    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "BMI": [bmi],
        "Systolic_BP": [systolic_bp],
        "Diastolic_BP": [diastolic_bp],
        "Cholesterol_mg_dL": [cholesterol],
        "HbA1c_percent": [hba1c],
        "Smoking_Status": [smoking],
        "Alcohol_Consumption_per_week": [alcohol],
        "Physical_Activity_hours_per_week": [activity],
        "Sleep_Hours_per_night": [sleep],
        "Avg_Heart_Rate": [heart_rate],
        "Daily_Steps": [steps],
        "Family_History_Diabetes": [family_diabetes],
        "Family_History_Heart_Disease": [family_heart]
    })

    prediction = model.predict(input_data)
    probs = model.predict_proba(input_data)
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    st.markdown("---")
    st.subheader("Assessment Result")

    if predicted_label == "Low":
        st.success("🟢 LOW RISK")
    elif predicted_label == "Medium":
        st.warning("🟡 MEDIUM RISK")
    elif predicted_label == "High":
        st.error("🔴 HIGH RISK")
    else:
        st.error("⚫ CRITICAL RISK")

    st.subheader("Risk Probability Distribution")
    prob_df = pd.DataFrame(probs, columns=label_encoder.classes_)
    st.bar_chart(prob_df.T)

    # PDF Report
    pdf_path = "patient_report.pdf"
    doc = SimpleDocTemplate(pdf_path)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("Healthcare Risk Report", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Predicted Risk: {predicted_label}", styles["Normal"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Model Accuracy: {accuracy:.2f}", styles["Normal"]))

    doc.build(elements)

    with open(pdf_path, "rb") as f:
        st.download_button("📥 Download Patient Report", f, file_name="Patient_Report.pdf")