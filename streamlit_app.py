import streamlit as st
import pandas as pd
import joblib
from PIL import Image
from datetime import datetime, timedelta
from xgboost import XGBClassifier
from sklearn.svm import SVC

# --- App Layout ---
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
st.title(" 🛠️ AI SMART MANUFACTURING PREDICTVE MAINTENANCE  👨‍🔧")
st.write(
    "Predict which component (comp1, comp2, comp3, comp4) will fail within the next 24 hours."
)

# --- Sidebar Menu ---
menu = st.sidebar.radio("Choose an option", [
    "🟦 Random Forest Model",
    "🟥 XGBoost Model",
    "🟨 Support Vector Machine Model",
    "🟢 Sensor Features Importance ",
    "🟣 Top 10 Predicted Failure",
    "🟤 Output in Excel"

])


# --- Helper to load and display images ---
def show_image(image_path, caption):
    try:
        img = Image.open(image_path)
        st.image(img, caption=caption, use_container_width =True)
    except:
        st.warning(f"⚠️ Image not found: {image_path}")

# --- Helper to load classification report ---
def show_text(report_path):
    try:
        with open(report_path, "r") as f:
            content = f.read()
        st.text(content)
    except:
        st.warning(f"⚠️ Report not found: {report_path}")

# -------------------------------
# 🟦 RANDOM FOREST MODEL
# -------------------------------
if menu == "🟦 Random Forest Model":
    
    st.subheader("🧪 Manual Prediction - Random Forest")
    rf_model = joblib.load("rf.pbz2")
    label_encoder = joblib.load("label_encoder.pkl")

    with st.form("rf_form"):
        machineID = st.number_input("Machine ID", value=1)
        volt = st.number_input("Voltage", value=160.0)
        rotate = st.number_input("Rotation", value=420.0)
        pressure = st.number_input("Pressure", value=110.0)
        vibration = st.number_input("Vibration", value=45.0)
        age = st.number_input("Age", value=10)

        submit = st.form_submit_button("Predict Failure")

        if submit:
            input_data = pd.DataFrame([[machineID, volt, rotate, pressure, vibration, age]],
                                  columns=['machineID', 'volt', 'rotate', 'pressure', 'vibration', 'age'])


            pred = rf_model.predict(input_data)
            predicted_label = label_encoder.inverse_transform(pred)[0]
            predicted_time = datetime.now() + timedelta(hours=24)

            st.success(f"🔧 Predicted Component Failure: **{predicted_label}**")

            if predicted_label != "none":
                predicted_time = datetime.now() + timedelta(hours=24)
                st.info(f"🕒 Recommended Maintenance Time: **{predicted_time.strftime('%Y-%m-%d %H:%M:%S')}**")
            else:
                st.info("🛠️ No maintenance needed at this time.")


    st.header("🟦 Random Forest Model Evaluation")
    show_image("confusion_matrix RF.png", "Confusion Matrix - Random Forest")
    show_image("class report rf.png", "Classification Report - Random Forest")

# -------------------------------
# 🟥 XGBoost Model
# -------------------------------
if menu == "🟥 XGBoost Model":
    
    st.subheader("🧪 Manual Prediction - XGBOOST")
    xgb_model = joblib.load("xgb_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    with st.form("xgb_form"):
        machineID = st.number_input("Machine ID", value=1)
        volt = st.number_input("Voltage", value=160.0)
        rotate = st.number_input("Rotation", value=420.0)
        pressure = st.number_input("Pressure", value=110.0)
        vibration = st.number_input("Vibration", value=45.0)
        age = st.number_input("Age", value=10)

        submit = st.form_submit_button("Predict Failure")

        if submit:
            input_data = pd.DataFrame([[machineID, volt, rotate, pressure, vibration, age]],
                                  columns=['machineID', 'volt', 'rotate', 'pressure', 'vibration', 'age'])


            pred = xgb_model.predict(input_data)
            predicted_label = label_encoder.inverse_transform(pred)[0]
            predicted_time = datetime.now() + timedelta(hours=24)

            st.success(f"🔧 Predicted Component Failure: **{predicted_label}**")

            if predicted_label != "none":
                predicted_time = datetime.now() + timedelta(hours=24)
                st.info(f"🕒 Recommended Maintenance Time: **{predicted_time.strftime('%Y-%m-%d %H:%M:%S')}**")
            else:
                st.info("🛠️ No maintenance needed at this time.")


    st.header("🟦 XGBoost Model Evaluation")
    show_image("confusion_matrix XGB.png", "Confusion Matrix - XGBoost")
    show_image("report XGB.png", "Classification Report - XGBoost")

# -------------------------------
# 🟨 Support Vector Machine Model
# -------------------------------
if menu == "🟨 Support Vector Machine Model":
    
    st.subheader("🧪 Manual Prediction - Support Vector Machine")
    svm_model = joblib.load("svm_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    with st.form("svm_form"):
        machineID = st.number_input("Machine ID", value=1)
        volt = st.number_input("Voltage", value=160.0)
        rotate = st.number_input("Rotation", value=420.0)
        pressure = st.number_input("Pressure", value=110.0)
        vibration = st.number_input("Vibration", value=45.0)
        age = st.number_input("Age", value=10)

        submit = st.form_submit_button("Predict Failure")

        if submit:
            input_data = pd.DataFrame([[machineID, volt, rotate, pressure, vibration, age]],
                                  columns=['machineID', 'volt', 'rotate', 'pressure', 'vibration', 'age'])


            pred = svm_model.predict(input_data)
            predicted_label = label_encoder.inverse_transform(pred)[0]
            predicted_time = datetime.now() + timedelta(hours=24)

            st.success(f"🔧 Predicted Component Failure: **{predicted_label}**")

            if predicted_label != "none":
                predicted_time = datetime.now() + timedelta(hours=24)
                st.info(f"🕒 Recommended Maintenance Time: **{predicted_time.strftime('%Y-%m-%d %H:%M:%S')}**")
            else:
                st.info("🛠️ No maintenance needed at this time.")


    st.header("🟦 SVM Model Evaluation")
    show_image("confusion_matrix SVM.png", "Confusion Matrix - SVM")
    show_image("Report SVM.png", "Classification Report - SVM")

# -------------------------------
# 🟢 SENSOR FEATURES IMPORTANCE
# -------------------------------
if menu == "🟢 Sensor Features Importance ":
    st.header("🟢 Feature Importance from Sensors")

    col1, col2 = st.columns(2)
    with col1:
        show_image("sensor feature importance.png", "Sensor Feature Importance")
    with col2:
        show_image("Shap_feature_importance_bar.png", "SHAP Feature Importance (Bar)")

        st.write(
            """Component Mapping:
            0: comp1  
            1: comp2  
            2: comp3  
            3: comp4  
            4: none"""
        )
# -------------------------------
# 🟣 TOP 10 PREDICTED FAILURE
# -------------------------------
elif menu == "🟣 Top 10 Predicted Failure":
    st.header("🟣 Top 10 Machines by Predicted Failure Frequency")
    show_image("top10_predicted_failures.png", "Top 10 Machines with Highest Predicted Failures")

# -------------------------------
# 🟤 OUTPUT IN EXCEL
# -------------------------------
elif menu == "🟤 Output in Excel":
    st.header("🟤 Predicted Failures - Excel Output")

    try:
        df = pd.read_csv("Predicted Failures.csv")
        st.dataframe(df)

        st.download_button(
            label="📥 Download Excel",
            data=df.to_csv(index=False),
            file_name="Predicted Failures.csv",
            mime='text/csv'
        )
    except:
        st.error("❌ Could not load 'Predicted Failures.csv'. Please ensure the file exists.")