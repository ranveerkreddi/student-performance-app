import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Load models and data
model = joblib.load("best_model.pkl")          # Random Forest: predicts final exam score
risk_model = joblib.load("at_risk_model.pkl")  # Logistic Regression: flags at-risk students (< 50 marks)
df = pd.read_excel("student_exam_prediction_dataset_extended copy.xlsx")

# Encode categorical variable for model
df['extracurricular_participation_encoded'] = df['extracurricular_participation'].map({'No': 0, 'Yes': 1})

features = ['study_hours_per_day', 'attendance_percentage', 'mental_health_rating', 'sleep_hours', 'extracurricular_participation_encoded']

# Page config
st.set_page_config(page_title="Student Performance Predictor", layout="wide")
st.title("Student Performance Prediction and Explainability")

with st.sidebar:
    st.header("Select Student and Adjust Inputs")
    student_ids = df["student_id"].unique()
    selected_id = st.selectbox("Select Student", student_ids)
    student_row = df[df["student_id"] == selected_id].iloc[0]

    study_hours = st.slider("Study Hours per Day", 0, 12, int(student_row["study_hours_per_day"]))
    attendance = st.slider("Attendance Percentage", 0, 100, int(student_row["attendance_percentage"]))
    sleep_hours = st.slider("Sleep Hours per Night", 0, 12, int(student_row["sleep_hours"]))
    mental_health = st.slider("Mental Health Rating (1-10)", 1, 10, int(student_row["mental_health_rating"]))
    extracurricular = st.selectbox(
        "Extracurricular Participation",
        ['No', 'Yes'],
        index=0 if student_row['extracurricular_participation'] == 'No' else 1
    )
    run_pred = st.button("Predict")

# Display Student Profile attractively
st.markdown("### Student Profile")
col1, col2 = st.columns(2)
with col1:
    st.write(f"**Age:** {student_row['age']}")
    st.write(f"**Gender:** {student_row['gender']}")
    st.write(f"**Program:** {student_row.get('program', 'N/A')}")
    st.write(f"**Diet Quality:** {student_row['diet_quality']}")
    st.write(f"**Exercise Frequency:** {student_row['exercise_frequency']}/week")
    st.write(f"**Extracurricular:** {student_row['extracurricular_participation']}")
with col2:
    st.write(f"**Study Hours:** {student_row['study_hours_per_day']} hrs/day")
    st.write(f"**Attendance:** {student_row['attendance_percentage']}%")
    st.write(f"**Sleep:** {student_row['sleep_hours']} hrs/night")
    st.write(f"**Mental Health:** {student_row['mental_health_rating']}/10")
    st.write(f"**Internet Quality:** {student_row['internet_quality']}")
    st.write(f"**Parental Education:** {student_row['parental_education_level']}")

score1, score2, score3, score4 = st.columns(4)
score1.metric("Python Score", f"{student_row['python_marks']}")
score2.metric("Mathematics", f"{student_row['mathematics_marks']}")
score3.metric("DBMS", f"{student_row['dbms_marks']}")
score4.metric("Final Exam", f"{student_row['final_exam_marks']}")

st.divider()

if run_pred:
    extra_enc = 1 if extracurricular == 'Yes' else 0
    input_data = np.array([[study_hours, attendance, mental_health, sleep_hours, extra_enc]])
    pred = model.predict(input_data)[0]
    pred = np.clip(pred, 0, 100)

    st.subheader("Prediction Result")
    st.metric("Predicted Final Exam Score", f"{pred:.2f}%")

    # ---- At-risk early warning (new) ----
    risk_df = pd.DataFrame(input_data, columns=features)
    risk_prob = risk_model.predict_proba(risk_df)[0, 1]
    st.subheader("Early-Warning Check")
    st.progress(float(risk_prob))
    if risk_prob >= 0.5:
        st.error(f"AT RISK: {risk_prob:.0%} probability of scoring below 50 marks. Consider early intervention.")
    elif risk_prob >= 0.25:
        st.warning(f"Watch closely: {risk_prob:.0%} probability of scoring below 50 marks.")
    else:
        st.success(f"Low risk: {risk_prob:.0%} probability of scoring below 50 marks.")
    st.caption("Risk model: balanced logistic regression, 5-fold CV recall ~90%, ROC-AUC ~0.96.")

    # SHAP explainability
    background = df[features].sample(100, random_state=42).values
    explainer = shap.TreeExplainer(model, background)
    shap_values = explainer.shap_values(input_data)
    shap_vals = shap_values[0] if isinstance(shap_values, list) else shap_values
    shap_vals = np.asarray(shap_vals).reshape(-1)  # works across shap versions

    colors = ['green' if val > 0 else 'red' for val in shap_vals]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(features, shap_vals, color=colors)
    ax.set_xlabel('Impact on Score')
    ax.set_title('Feature Impact on Predicted Score')
    st.pyplot(fig)

    expl_text = []
    for feat, val in zip(features, shap_vals):
        expl_text.append(
            f"- Higher **{feat.replace('_', ' ')}** {'increases' if val > 0 else 'decreases'} predicted score by {abs(val):.2f} points."
        )
    st.markdown("### Explanation Summary\n" + "\n".join(expl_text))
  
