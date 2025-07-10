import streamlit as st
import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Parkinson's Disease Prediction", layout="wide")

# Injecting custom CSS for font and style
st.markdown("""
    <style>
    label, .stSlider label, .stSelectbox label {
        font-weight: bold !important;
        font-size: 22px !important;
        color: #333333 !important;
    }

    .stSlider > div, .stSelectbox > div {
        font-size: 20px !important;
    }

    .stSlider, .stSelectbox {
        margin-bottom: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🧠 Parkinson's Disease Prediction Dashboard")
st.markdown("### Predict the likelihood of Parkinson's Disease using patient-specific features.")

# Load model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

# Feature list
feature_names = [
    'Age', 'Gender', 'FamilyHistoryParkinsons', 'TraumaticBrainInjury', 'UPDRS',
    'MoCA', 'FunctionalAssessment', 'Tremor', 'Rigidity', 'Bradykinesia',
    'PosturalInstability', 'SpeechProblems', 'SleepDisorders', 'Constipation'
]

# Input section
st.markdown("## 🧾 Enter Patient Information")

st.markdown("<h4 style='font-weight: bold; font-size: 22px;'>Age</h4>", unsafe_allow_html=True)
age = st.slider('', 50, 90, 70, key='age')

st.markdown("<h4 style='font-weight: bold; font-size: 22px;'>Gender</h4>", unsafe_allow_html=True)
gender = st.selectbox('', [0, 1], format_func=lambda x: 'Male' if x == 0 else 'Female', key='gender')

st.markdown("<h4 style='font-weight: bold; font-size: 22px;'>Family History of Parkinson's</h4>", unsafe_allow_html=True)
family_history = st.selectbox('', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', key='family_history')

st.markdown("<h4 style='font-weight: bold; font-size: 22px;'>History of Traumatic Brain Injury</h4>", unsafe_allow_html=True)
tbi = st.selectbox('', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', key='tbi')

st.markdown("<h4 style='font-weight: bold; font-size: 22px;'>UPDRS (0–199)</h4>", unsafe_allow_html=True)
updrs = st.slider('', 0, 199, 50, key='updrs')

st.markdown("<h4 style='font-weight: bold; font-size: 22px;'>MoCA Score (0–30)</h4>", unsafe_allow_html=True)
moca = st.slider('', 0, 30, 15, key='moca')

st.markdown("<h4 style='font-weight: bold; font-size: 22px;'>Functional Assessment (0–10)</h4>", unsafe_allow_html=True)
func_assess = st.slider('', 0, 10, 5, key='func_assess')

st.markdown("<h4 style='font-weight: bold; font-size: 22px;'>Tremor</h4>", unsafe_allow_html=True)
tremor = st.selectbox('', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', key='tremor')

st.markdown("<h4 style='font-weight: bold; font-size: 22px;'>Rigidity</h4>", unsafe_allow_html=True)
rigidity = st.selectbox('', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', key='rigidity')

st.markdown("<h4 style='font-weight: bold; font-size: 22px;'>Bradykinesia</h4>", unsafe_allow_html=True)
brady = st.selectbox('', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', key='brady')

st.markdown("<h4 style='font-weight: bold; font-size: 22px;'>Postural Instability</h4>", unsafe_allow_html=True)
posture = st.selectbox('', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', key='posture')

st.markdown("<h4 style='font-weight: bold; font-size: 22px;'>Speech Problems</h4>", unsafe_allow_html=True)
speech = st.selectbox('', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', key='speech')

st.markdown("<h4 style='font-weight: bold; font-size: 22px;'>Sleep Disorders</h4>", unsafe_allow_html=True)
sleep = st.selectbox('', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', key='sleep')

st.markdown("<h4 style='font-weight: bold; font-size: 22px;'>Constipation</h4>", unsafe_allow_html=True)
constip = st.selectbox('', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No', key='constip')


# Collecting input
user_input = {
    'Age': age,
    'Gender': gender,
    'FamilyHistoryParkinsons': family_history,
    'TraumaticBrainInjury': tbi,
    'UPDRS': updrs,
    'MoCA': moca,
    'FunctionalAssessment': func_assess,
    'Tremor': tremor,
    'Rigidity': rigidity,
    'Bradykinesia': brady,
    'PosturalInstability': posture,
    'SpeechProblems': speech,
    'SleepDisorders': sleep,
    'Constipation': constip
}

input_df = pd.DataFrame([user_input], columns=feature_names)

# Prediction button
if st.button("🧪 Detect Parkinson's Disease"):

    # Mapping for display names and binary values
    label_mapping = {
        "Gender": {0: "Male", 1: "Female"},
        "FamilyHistoryParkinsons": {0: "No", 1: "Yes"},
        "TraumaticBrainInjury": {0: "No", 1: "Yes"},
        "Tremor": {0: "No", 1: "Yes"},
        "Rigidity": {0: "No", 1: "Yes"},
        "Bradykinesia": {0: "No", 1: "Yes"},
        "PosturalInstability": {0: "No", 1: "Yes"},
        "SpeechProblems": {0: "No", 1: "Yes"},
        "SleepDisorders": {0: "No", 1: "Yes"},
        "Constipation": {0: "No", 1: "Yes"},
    }

    display_labels = {
        "Age": "Age",
        "Gender": "Gender",
        "FamilyHistoryParkinsons": "Family History Parkinson's",
        "TraumaticBrainInjury": "Traumatic Brain Injury",
        "UPDRS": "UPDRS",
        "MoCA": "MoCA",
        "FunctionalAssessment": "Functional Assessment",
        "Tremor": "Tremor",
        "Rigidity": "Rigidity",
        "Bradykinesia": "Bradykinesia",
        "PosturalInstability": "Postural Instability",
        "SpeechProblems": "Speech Problems",
        "SleepDisorders": "Sleep Disorders",
        "Constipation": "Constipation"
    }

    st.markdown("## 🧾 Entered Patient Features")
    for key, value in user_input.items():
        display_label = display_labels.get(key, key)
        if key in label_mapping:
            readable_value = label_mapping[key].get(value, value)
        else:
            readable_value = value
        st.markdown(f"<p style='font-size:18px'><strong>{display_label}:</strong> {readable_value}</p>", unsafe_allow_html=True)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("## 🩺 Prediction Result")

    # Styled diagnosis output
    if prediction == 1:
        st.markdown(
            f"""
            <div style="background-color:#ffe6e6;padding:20px;border-radius:10px;border-left:6px solid #d32f2f;">
                <h3 style="color:#b71c1c;">✅ Diagnosis: <strong>Positive for Parkinson's Disease</strong></h3>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="background-color:#e7f3fe;padding:20px;border-radius:10px;border-left:6px solid #2e7d32;">
                <h3 style="color:#1b5e20;">❎ Diagnosis: <strong>Negative for Parkinson's Disease</strong></h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Styled probability/confidence
    st.markdown(
        f"""
        <div style="margin-top:20px;padding:15px 20px;background-color:#f9f9f9;border-radius:8px;border:1px solid #ccc;">
            <h4 style="font-size:22px;color:#333;">Prediction Confidence:</h4>
            <p style="font-size:26px;font-weight:bold;color:#007acc;">{probability:.2f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # SHAP values
    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)

    if shap_values.values.ndim == 3:
        shap_1d = shap_values.values[0, :, 1]
    else:
        raise ValueError(f"Unexpected SHAP shape: {shap_values.values.shape}")

    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'Contribution': shap_1d
    }).sort_values(by='Contribution', key=lambda x: abs(x), ascending=False)

    st.markdown("## 🔍 Top Feature Contributions (SHAP)")

    # Prepare SHAP data as % and style by sign
    shap_df_display = shap_df.head(10).copy()
    shap_df_display['Contribution (%)'] = shap_df_display['Contribution'] * 100

    # Format each row based on positive/negative sign
    def format_contribution(value):
        color = 'red' if value >= 0 else 'blue'
        return f"<span style='color:{color}; font-weight:bold;'>{value:.2f}%</span>"

    shap_df_display['Contribution (%)'] = shap_df_display['Contribution (%)'].apply(format_contribution)

    # Building styled HTML table
    shap_html_table = shap_df_display[['Feature', 'Contribution (%)']].to_html(
        escape=False, index=False, classes='shap-table'
    )

    # Injecting CSS
    st.markdown("""
        <style>
        .shap-table {
            font-size: 20px;
            font-weight: 500;
            border-collapse: collapse;
            width: 100%;
            margin-top: 20px;
        }
        .shap-table th {
            background-color: #f2f2f2;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }
        .shap-table td {
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(shap_html_table, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:10px; font-size:16px;">
        <strong style="color:red;">Red</strong>: Feature is contributing <em>toward</em> a Parkinson's diagnosis<br>
        <strong style="color:blue;">Blue</strong>: Feature is contributing <em>against</em> a Parkinson's diagnosis
    </div>
    """, unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x='Contribution', y='Feature', data=shap_df.head(10), palette='coolwarm', ax=ax)
    ax.set_title("Top Features Influencing Prediction")
    st.pyplot(fig)

st.caption("🔬 Built by Afsha Anjum | 💡 Powered by Streamlit")
