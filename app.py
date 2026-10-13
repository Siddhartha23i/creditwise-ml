from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "loan_approval_data.csv"
MODEL_PATH = BASE_DIR / "creditwise_selected_model.joblib"
METRICS_PATH = BASE_DIR / "model_metrics.json"
FEATURE_ORDER = [
    "Applicant_Income",
    "Coapplicant_Income",
    "Employment_Status",
    "Age",
    "Marital_Status",
    "Dependents",
    "Credit_Score",
    "Existing_Loans",
    "DTI_Ratio",
    "Savings",
    "Collateral_Value",
    "Loan_Amount",
    "Loan_Term",
    "Loan_Purpose",
    "Property_Area",
    "Education_Level",
    "Gender",
    "Employer_Category",
]


@st.cache_resource
def load_or_train_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    from train_model import train_model

    model, _ = train_model()
    return model


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


def make_prediction_row(values: dict) -> pd.DataFrame:
    row = {name: values.get(name, "") for name in FEATURE_ORDER}
    return pd.DataFrame([row])


def get_feature_explanation(model):
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    transformed_names = preprocessor.get_feature_names_out().tolist()
    if hasattr(classifier, "coef_"):
        values = np.abs(classifier.coef_[0])
        label = "absolute coefficient"
    elif hasattr(classifier, "theta_"):
        values = np.abs(classifier.theta_[1] - classifier.theta_[0])
        label = "class mean separation"
    else:
        return None, None

    explanation = pd.DataFrame({"feature": transformed_names, "value": values})
    return explanation.sort_values("value", ascending=False).head(10), label


def predict_frame(model, frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    probabilities = model.predict_proba(frame)[:, 1]
    approvals = pd.Series(np.where(probabilities >= 0.5, "Yes", "No"), index=frame.index)
    return probabilities, approvals


def render_single_prediction(model) -> None:
    with st.form("loan_form"):
        col1, col2 = st.columns(2)

        with col1:
            applicant_income = st.number_input("Applicant income", min_value=0.0, value=25000.0, step=500.0)
            coapplicant_income = st.number_input("Co-applicant income", min_value=0.0, value=1500.0, step=500.0)
            employment_status = st.selectbox(
                "Employment status",
                ["Salaried", "Self-employed", "Contract", "Unemployed"],
            )
            age = st.number_input("Age", min_value=18, max_value=80, value=35, step=1)
            marital_status = st.selectbox("Marital status", ["Single", "Married", "Divorced", "Widowed"])
            dependents = st.number_input("Dependents", min_value=0, max_value=10, value=1, step=1)
            credit_score = st.number_input("Credit score", min_value=300, max_value=850, value=680, step=1)
            existing_loans = st.number_input("Existing loans", min_value=0, max_value=20, value=2, step=1)
            dti_ratio = st.number_input("DTI ratio", min_value=0.0, max_value=1.0, value=0.35, step=0.01)

        with col2:
            savings = st.number_input("Savings", min_value=0.0, value=12000.0, step=500.0)
            collateral_value = st.number_input("Collateral value", min_value=0.0, value=35000.0, step=500.0)
            loan_amount = st.number_input("Loan amount", min_value=1000.0, value=15000.0, step=500.0)
            loan_term = st.number_input("Loan term (months)", min_value=6, max_value=240, value=60, step=6)
            loan_purpose = st.selectbox(
                "Loan purpose",
                ["Home", "Car", "Education", "Business", "Personal"],
            )
            property_area = st.selectbox("Property area", ["Urban", "Semiurban", "Rural"])
            education_level = st.selectbox("Education level", ["Graduate", "Not Graduate"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            employer_category = st.selectbox(
                "Employer category",
                ["Private", "Government", "MNC", "Business", "Unemployed"],
            )

        submitted = st.form_submit_button("Predict approval")

    if submitted:
        values = {
            "Applicant_Income": applicant_income,
            "Coapplicant_Income": coapplicant_income,
            "Employment_Status": employment_status,
            "Age": age,
            "Marital_Status": marital_status,
            "Dependents": dependents,
            "Credit_Score": credit_score,
            "Existing_Loans": existing_loans,
            "DTI_Ratio": dti_ratio,
            "Savings": savings,
            "Collateral_Value": collateral_value,
            "Loan_Amount": loan_amount,
            "Loan_Term": loan_term,
            "Loan_Purpose": loan_purpose,
            "Property_Area": property_area,
            "Education_Level": education_level,
            "Gender": gender,
            "Employer_Category": employer_category,
        }
        prediction_frame = make_prediction_row(values)
        probability = float(model.predict_proba(prediction_frame)[0, 1])
        approval = "Approved" if probability >= 0.5 else "Rejected"

        st.subheader("Prediction")
        st.metric("Approval probability", f"{probability * 100:.1f}%")

        if approval == "Approved":
            st.success(f"Recommendation: {approval}")
        else:
            st.warning(f"Recommendation: {approval}")

        if probability >= 0.7:
            st.info("Strong likelihood of approval based on the modeled applicant profile.")
        elif probability <= 0.3:
            st.info("The applicant profile suggests a higher risk of rejection.")
        else:
            st.info("The application is near the decision threshold and should be reviewed manually.")

        explanation_df, explanation_label = get_feature_explanation(model)
        if explanation_df is not None:
            st.subheader("Top model signals")
            st.caption(f"Shown using {explanation_label}; this is not a causal explanation.")
            st.bar_chart(explanation_df.set_index("feature")["value"])


def render_batch_prediction(model) -> None:
    uploaded = st.file_uploader("Upload a CSV with applicant records", type=["csv"])
    if uploaded is None:
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception:
        st.error("The uploaded file is not a valid CSV. Please check the format and try again.")
        return

    missing = [col for col in FEATURE_ORDER if col not in df.columns]
    if missing:
        st.error(f"The uploaded file is missing required columns: {', '.join(missing)}")
        return

    prediction_frame = df[FEATURE_ORDER].copy()
    probabilities, approvals = predict_frame(model, prediction_frame)
    results = df.copy()
    results["Approval_Probability"] = probabilities
    results["Decision"] = approvals

    st.subheader("Batch prediction results")
    st.dataframe(results, use_container_width=True)

    accepted = (results["Decision"] == "Yes").sum()
    st.metric("Likely approvals", int(accepted))
    st.metric("Approval rate", f"{(accepted / len(results) * 100):.1f}%")


def main() -> None:
    st.set_page_config(page_title="CreditWise ML", page_icon="💳", layout="wide")
    st.title("CreditWise ML")
    st.caption("Loan approval prediction for new applicants")

    df = load_dataset()
    labeled_count = df["Loan_Approved"].notna().sum()
    unlabeled_count = df["Loan_Approved"].isna().sum()

    with st.sidebar:
        st.subheader("Dataset overview")
        st.metric("Total applications", len(df))
        st.metric("Labeled", int(labeled_count))
        st.metric("Unlabeled", int(unlabeled_count))

        st.markdown("---")
        st.subheader("Prediction mode")
        mode = st.radio("Choose one", ["Single applicant", "CSV upload"], index=0)

    st.subheader("Loan approval model")
    if METRICS_PATH.exists():
        metadata = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        selected_model = metadata.get("selected_model", "selected permitted classifier")
    else:
        selected_model = "selected permitted classifier"
    st.write(
        f"This app uses {selected_model} trained on historical loan applications to estimate approval probability. "
        "This is an educational prediction, not a real lending decision."
    )

    model = load_or_train_model()
    sample = df.dropna(subset=["Loan_Approved"]).head(10)
    st.dataframe(sample, use_container_width=True)

    if mode == "Single applicant":
        render_single_prediction(model)
    else:
        render_batch_prediction(model)


if __name__ == "__main__":
    main()
