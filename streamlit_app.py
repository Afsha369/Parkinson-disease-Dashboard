import streamlit as st
import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page config FIRST
st.set_page_config(page_title="Parkinson's Disease Prediction", layout="wide")

# Title and Description
st.title("🧠 Parkinson's Disease Prediction Dashboard")
st.write("Predict likelihood of Parkinson's Disease using patient features.")

# Load trained Random Forest model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

# Feature list used in the model
feature_names = [
    'Age', 'Gender', 'FamilyHistoryParkinsons', 'TraumaticBrainInjury', 'UPDRS',
    'MoCA', 'FunctionalAssessment', 'Tremor', 'Rigidity', 'Bradykinesia',
    'PosturalInstability', 'SpeechProblems', 'SleepDisorders', 'Constipation'
]

# Sidebar for patient input
st.sidebar.header("Patient Input Features")

def get_user_input():
    return {
        'Age': st.sidebar.slider('Age', 50, 90, 70),
        'Gender': st.sidebar.selectbox('Gender', [0, 1], format_func=lambda x: 'Male' if x == 0 else 'Female'),
        'FamilyHistoryParkinsons': st.sidebar.selectbox("Family History of Parkinson's", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No'),
        'TraumaticBrainInjury': st.sidebar.selectbox('History of Traumatic Brain Injury', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No'),
        'UPDRS': st.sidebar.slider('UPDRS (0-199)', 0, 199, 50),
        'MoCA': st.sidebar.slider('MoCA Score (0-30)', 0, 30, 15),
        'FunctionalAssessment': st.sidebar.slider('Functional Assessment (0-10)', 0, 10, 5),
        'Tremor': st.sidebar.selectbox('Tremor', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No'),
        'Rigidity': st.sidebar.selectbox('Rigidity', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No'),
        'Bradykinesia': st.sidebar.selectbox('Bradykinesia', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No'),
        'PosturalInstability': st.sidebar.selectbox('Postural Instability', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No'),
        'SpeechProblems': st.sidebar.selectbox('Speech Problems', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No'),
        'SleepDisorders': st.sidebar.selectbox('Sleep Disorders', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No'),
        'Constipation': st.sidebar.selectbox('Constipation', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No'),
    }

user_input = get_user_input()
input_df = pd.DataFrame([user_input], columns=feature_names)

# Prediction and Results
if st.button("Predict Parkinson's Diagnosis"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"Predicted Diagnosis: **Positive for Parkinson's**")
    else:
        st.info(f"Predicted Diagnosis: **Negative for Parkinson's**")

    st.write(f"**Probability of Parkinson's:** {probability:.2f}")


    # ---- SHAP EXPLAINING SECTION ----
    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)

    # SHAP generates shape: (1, features, classes)
    if shap_values.values.ndim == 3:
        # Select class 1 (Positive Parkinson’s) explanation
        shap_1d = shap_values.values[0, :, 1]
    else:
        raise ValueError(f"Unexpected SHAP shape: {shap_values.values.shape}")

    shap_feature_names = feature_names

    # Verify lengths
    assert len(shap_feature_names) == shap_1d.shape[0], "Feature length mismatch."

    shap_df = pd.DataFrame({
        'Feature': shap_feature_names,
        'Contribution': shap_1d
    }).sort_values(by='Contribution', key=lambda x: abs(x), ascending=False)

    # Display nicely in a table
    st.subheader("🔎 Feature Contributions for Positive Prediction")
    st.dataframe(shap_df.head(10).style.format({'Contribution': '{:.4f}'}))

    # Optional horizontal bar plot
    st.subheader("📊 Feature Impact (Bar Chart)")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x='Contribution', y='Feature', data=shap_df.head(10), palette='coolwarm', ax=ax)
    ax.set_title("Top Feature Contributions to Parkinson’s Prediction")
    st.pyplot(fig)

st.caption("Built by Afsha Anjum | Powered by Streamlit")