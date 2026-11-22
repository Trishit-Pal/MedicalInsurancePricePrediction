import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

# Same IQR limits used to cap BMI in MLHandsOn.ipynb.
BMI_MIN = 13.675
BMI_MAX = 47.315

st.set_page_config(page_title="Medical Insurance Cost Predictor", layout="centered")


@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "insurancemodelf.pkl"
    with open(model_path, "rb") as f:
        return pickle.load(f)


model = load_model()

st.title("Medical Insurance Cost Predictor")
st.write("Enter basic health details to estimate yearly medical insurance charges.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 64, 30)
        children = st.slider("Children", 0, 5, 0)
    with col2:
        bmi = st.number_input("BMI", min_value=15.0, max_value=55.0, value=28.0, step=0.1)
        smokes = st.selectbox("Smoker", ["No", "Yes"])

    submitted = st.form_submit_button("Estimate cost", type="primary")

if submitted:
    capped_bmi = min(max(bmi, BMI_MIN), BMI_MAX)
    row = pd.DataFrame([{
        "age": age,
        "bmi": capped_bmi,
        "children": children,
        "smoker": 0 if smokes == "Yes" else 1,  # 0 = yes, 1 = no (matches notebook)
    }])
    cost = max(0.0, float(model.predict(row)[0]))
    st.metric("Estimated yearly charge", f"${cost:,.2f}")
    st.caption("Learning project based on insurance.csv, not an actual quote. Sex and region are not used.")
