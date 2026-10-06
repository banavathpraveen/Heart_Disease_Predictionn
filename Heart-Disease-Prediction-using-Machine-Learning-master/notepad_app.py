import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

st.title("Heart Disease Prediction")
st.write("Fill in the patient parameters below to predict heart disease risk.")

# Input widgets
age = st.slider("Age", 1, 120, 30)
sex = st.selectbox("Sex (0=Female,1=Male)", [0,1])
cp = st.selectbox("Chest Pain Type (cp)", [0,1,2,3])
trestbps = st.slider("Resting Blood Pressure (trestbps)", 50, 250, 120)
chol = st.slider("Cholesterol (chol)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl (fbs)", [0,1])
restecg = st.selectbox("Resting ECG (restecg)", [0,1,2])
thalach = st.slider("Maximum Heart Rate Achieved (thalach)", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina (exang)", [0,1])
oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0, 0.1)
slope = st.selectbox("Slope of ST Segment (slope)", [0,1,2])
ca = st.selectbox("Number of major vessels colored by fluoroscopy (ca)", [0,1,2,3,4])
thal = st.selectbox("Thalassemia (thal)", [0,1,2,3])

# Model selection
model_choice = st.selectbox("Select Model", ["Random Forest", "AdaBoost", "XGBoost", "Neural Network"])

if st.button("Predict Heart Disease"):
    input_data = pd.DataFrame({
        "age":[age], "sex":[sex], "cp":[cp], "trestbps":[trestbps],
        "chol":[chol], "fbs":[fbs], "restecg":[restecg], "thalach":[thalach],
        "exang":[exang], "oldpeak":[oldpeak], "slope":[slope], "ca":[ca], "thal":[thal]
    })

    # Load models (make sure they are saved in project folder)
    rf_model = joblib.load("rf_model.pkl")
    ab_model = joblib.load("ab_model.pkl")
    xgb_model = joblib.load("xgb_model.pkl")
    nn_model = load_model("nn_model.h5")

    # Predict
    if model_choice == "Random Forest":
        pred = rf_model.predict(input_data)[0]
    elif model_choice == "AdaBoost":
        pred = ab_model.predict(input_data)[0]
    elif model_choice == "XGBoost":
        pred = xgb_model.predict(input_data)[0]
    elif model_choice == "Neural Network":
        pred_prob = nn_model.predict(input_data)
        pred = 1 if pred_prob[0][0] > 0.5 else 0

    if pred == 1:
        st.error("Warning: High risk of heart disease!")
    else:
        st.success("Low risk of heart disease!")
