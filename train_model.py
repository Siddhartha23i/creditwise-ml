from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "loan_approval_data.csv"
MODEL_PATH = BASE_DIR / "creditwise_model.joblib"
PREDICTIONS_PATH = BASE_DIR / "predictions.csv"


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df


def build_pipeline() -> Pipeline:
    numeric_features = [
        "Applicant_Income",
        "Coapplicant_Income",
        "Age",
        "Dependents",
        "Credit_Score",
        "Existing_Loans",
        "DTI_Ratio",
        "Savings",
        "Collateral_Value",
        "Loan_Amount",
        "Loan_Term",
    ]
    categorical_features = [
        "Employment_Status",
        "Marital_Status",
        "Loan_Purpose",
        "Property_Area",
        "Education_Level",
        "Gender",
        "Employer_Category",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    classifier = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        class_weight="balanced",
    )

    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def prepare_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    labeled_df = df.dropna(subset=["Loan_Approved"]).copy()
    labeled_df["Loan_Approved"] = (
        labeled_df["Loan_Approved"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    )
    labeled_df = labeled_df.drop(columns=["Applicant_ID"], errors="ignore")
    features = labeled_df.drop(columns=["Loan_Approved"])
    target = labeled_df["Loan_Approved"].astype(int)
    return features, target


def train_model() -> tuple[Pipeline, dict]:
    df = load_dataset()
    features, target = prepare_training_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    predicted = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predicted)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probability)), 4),
        "report": classification_report(y_test, predicted, target_names=["No", "Yes"]),
    }

    joblib.dump(model, MODEL_PATH)

    unlabeled = df[df["Loan_Approved"].isna()].copy()
    if not unlabeled.empty:
        unlabeled_features = unlabeled.drop(columns=["Applicant_ID", "Loan_Approved"], errors="ignore")
        probabilities = model.predict_proba(unlabeled_features)[:, 1]
        unlabeled["Predicted_Approval_Probability"] = probabilities
        unlabeled["Predicted_Approval"] = np.where(probabilities >= 0.5, "Yes", "No")
        unlabeled.to_csv(PREDICTIONS_PATH, index=False)

    return model, metrics


if __name__ == "__main__":
    _, metrics = train_model()
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"ROC AUC: {metrics['roc_auc']}")
    print(metrics["report"])
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved predictions: {PREDICTIONS_PATH}")
