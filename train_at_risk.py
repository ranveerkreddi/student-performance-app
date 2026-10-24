"""
Train and evaluate an at-risk student classifier.

Run from the repo folder (next to the dataset):
    python train_at_risk.py

Label: a student is "at risk" if final_exam_marks < 50.
Features: the same 5 inputs the Streamlit app already uses.
Outputs: cross-validated metrics, a held-out test report, and at_risk_model.pkl
"""
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_FILE = "student_exam_prediction_dataset_extended copy.xlsx"
THRESHOLD_MARKS = 50
FEATURES = [
    "study_hours_per_day",
    "attendance_percentage",
    "mental_health_rating",
    "sleep_hours",
    "extracurricular_participation_encoded",
]

df = pd.read_excel(DATA_FILE)
df["extracurricular_participation_encoded"] = df["extracurricular_participation"].map({"No": 0, "Yes": 1})

X = df[FEATURES].fillna(df[FEATURES].median())
y = (df["final_exam_marks"] < THRESHOLD_MARKS).astype(int)  # 1 = at risk

print(f"Students: {len(df)} | at risk: {y.sum()} ({y.mean():.1%}) | not at risk: {(1 - y).sum()}")
print(f"Accuracy of always predicting 'not at risk' (baseline): {1 - y.mean():.1%}\n")

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(class_weight="balanced", max_iter=1000),
)

# 1) 5-fold stratified cross-validation (most reliable number for a small dataset)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
print("5-fold cross-validation (mean):")
for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
    print(f"  {metric:10s}: {scores['test_' + metric].mean():.3f}")

# 2) Held-out test set report
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]
print("\nHeld-out test set (20%):")
print(classification_report(y_test, pred, target_names=["not at risk", "at risk"], digits=3))
print("Confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_test, pred))
print(f"ROC-AUC: {roc_auc_score(y_test, proba):.3f}")

# 3) Fit on all data for the app and save
model.fit(X, y)
joblib.dump(model, "at_risk_model.pkl")
coefs = dict(zip(FEATURES, np.round(model[-1].coef_[0], 3)))
print("\nStandardised coefficients (positive = raises risk):", coefs)
print("Saved at_risk_model.pkl")