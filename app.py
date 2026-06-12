import streamlit as st
import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide"
)

# ==========================================
# LOAD FILES
# ==========================================

model = load_model("models/placement_ann.keras")
scaler = joblib.load("models/scaler.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.stButton > button {
    width: 100%;
    height: 3em;
    font-size: 18px;
    font-weight: bold;
}

.metric-box {
    background-color: #f0f2f6;
    padding: 15px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================

st.title("🎓 Student Placement Prediction System")
st.markdown(
    "Artificial Neural Network (ANN) based prediction of student placement opportunities."
)

st.divider()

# ==========================================
# INPUT SECTION
# ==========================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("📚 Academic Details")

    gender = st.selectbox(
        "Gender",
        label_encoders["Gender"].classes_
    )

    board10 = st.selectbox(
        "10th Board",
        label_encoders["10th board"].classes_
    )

    marks10 = st.slider(
        "10th Marks (%)",
        0.0,
        100.0,
        80.0
    )

    board12 = st.selectbox(
        "12th Board",
        label_encoders["12th board"].classes_
    )

    marks12 = st.slider(
        "12th Marks (%)",
        0.0,
        100.0,
        75.0
    )

    stream = st.selectbox(
        "Stream",
        label_encoders["Stream"].classes_
    )

    cgpa = st.slider(
        "CGPA",
        0.0,
        10.0,
        8.0
    )

with col2:

    st.subheader("💼 Skills & Experience")

    internship = st.selectbox(
        "Internship",
        label_encoders["Internships(Y/N)"].classes_
    )

    training = st.selectbox(
        "Training",
        label_encoders["Training(Y/N)"].classes_
    )

    backlog = st.selectbox(
        "Backlog in 5th Sem",
        label_encoders["Backlog in 5th sem"].classes_
    )

    project = st.selectbox(
        "Innovative Project",
        label_encoders["Innovative Project(Y/N)"].classes_
    )

    communication = st.slider(
        "Communication Level",
        1,
        5,
        3
    )

    technical = st.selectbox(
        "Technical Course",
        label_encoders["Technical Course(Y/N)"].classes_
    )

st.divider()

# ==========================================
# PREDICT BUTTON
# ==========================================

if st.button("🚀 Predict Placement"):

    input_df = pd.DataFrame([{

        "Gender":
        label_encoders["Gender"].transform([gender])[0],

        "10th board":
        label_encoders["10th board"].transform([board10])[0],

        "10th marks":
        marks10,

        "12th board":
        label_encoders["12th board"].transform([board12])[0],

        "12th marks":
        marks12,

        "Stream":
        label_encoders["Stream"].transform([stream])[0],

        "Cgpa":
        cgpa,

        "Internships(Y/N)":
        label_encoders["Internships(Y/N)"].transform([internship])[0],

        "Training(Y/N)":
        label_encoders["Training(Y/N)"].transform([training])[0],

        "Backlog in 5th sem":
        label_encoders["Backlog in 5th sem"].transform([backlog])[0],

        "Innovative Project(Y/N)":
        label_encoders["Innovative Project(Y/N)"].transform([project])[0],

        "Communication level":
        communication,

        "Technical Course(Y/N)":
        label_encoders["Technical Course(Y/N)"].transform([technical])[0]

    }])

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled, verbose=0)

    probability = float(prediction[0][0])

    st.divider()

    metric1, metric2 = st.columns(2)

    with metric1:

        st.metric(
            "Placement Probability",
            f"{probability*100:.2f}%"
        )

    with metric2:

        st.metric(
            "Communication Level",
            communication
        )

    st.progress(probability)

    if probability >= 0.5:

        st.success(
            f"✅ Student is likely to be PLACED\n\nConfidence: {probability*100:.2f}%"
        )

    else:

        st.error(
            f"❌ Student is less likely to be PLACED\n\nConfidence: {(1-probability)*100:.2f}%"
        )

    st.subheader("📋 Student Summary")

    st.write({
        "CGPA": cgpa,
        "10th Marks": marks10,
        "12th Marks": marks12,
        "Communication Level": communication,
        "Internship": internship,
        "Training": training,
        "Technical Course": technical
    })

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Developed using TensorFlow, Scikit-Learn and Streamlit"
)