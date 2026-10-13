from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "loan_approval_data.csv"
MODEL_PATH = BASE_DIR / "creditwise_selected_model.joblib"
METRICS_PATH = BASE_DIR / "model_metrics.json"
PREDICTIONS_PATH = BASE_DIR / "predictions.csv"

NUMERIC_FEATURES = [
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
CATEGORICAL_FEATURES = [
    "Employment_Status",
    "Marital_Status",
    "Loan_Purpose",
    "Property_Area",
    "Education_Level",
    "Gender",
    "Employer_Category",
]


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def build_model_pipelines() -> dict[str, Pipeline]:
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Gaussian Naive Bayes": GaussianNB(),
    }
    return {
        name: Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                ("classifier", classifier),
            ]
        )
        for name, classifier in models.items()
    }


def prepare_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    labeled_df = df.dropna(subset=["Loan_Approved"]).copy()
    labeled_df["Loan_Approved"] = (
        labeled_df["Loan_Approved"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
    )
    labeled_df = labeled_df.dropna(subset=["Loan_Approved"])
    features = labeled_df.drop(columns=["Applicant_ID", "Loan_Approved"], errors="ignore")
    target = labeled_df["Loan_Approved"].astype(int)
    return features, target


def evaluate_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]
    matrix = confusion_matrix(y_test, predicted, labels=[0, 1])

    return {
        "accuracy": round(float(accuracy_score(y_test, predicted)), 4),
        "precision": round(float(precision_score(y_test, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predicted, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predicted, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probability)), 4),
        "confusion_matrix": matrix.tolist(),
    }


def select_deployment_model(metrics: dict[str, dict]) -> str:
    # ROC AUC is the primary criterion because the app exposes probabilities;
    # F1 and accuracy provide deterministic tie-breakers.
    return max(
        metrics,
        key=lambda name: (
            metrics[name]["roc_auc"],
            metrics[name]["f1"],
            metrics[name]["accuracy"],
        ),
    )


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

    model_pipelines = build_model_pipelines()
    metrics = {
        name: evaluate_model(model, X_train, X_test, y_train, y_test)
        for name, model in model_pipelines.items()
    }
    selected_name = select_deployment_model(metrics)
    selected_model = model_pipelines[selected_name]

    joblib.dump(selected_model, MODEL_PATH)
    metadata = {
        "selected_model": selected_name,
        "selection_criterion": "Highest ROC AUC; F1 and accuracy used as tie-breakers",
        "random_state": 42,
        "metrics": metrics,
    }
    METRICS_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    unlabeled = df[df["Loan_Approved"].isna()].copy()
    if not unlabeled.empty:
        unlabeled_features = unlabeled.drop(
            columns=["Applicant_ID", "Loan_Approved"],
            errors="ignore",
        )
        probabilities = selected_model.predict_proba(unlabeled_features)[:, 1]
        unlabeled["Predicted_Approval_Probability"] = probabilities
        unlabeled["Predicted_Approval"] = np.where(probabilities >= 0.5, "Yes", "No")
        unlabeled.to_csv(PREDICTIONS_PATH, index=False)

    return selected_model, metadata


if __name__ == "__main__":
    _, metadata = train_model()
    print(f"Selected deployment model: {metadata['selected_model']}")
    print(f"Selection criterion: {metadata['selection_criterion']}")
    for name, values in metadata["metrics"].items():
        print(f"{name}: {values}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved predictions: {PREDICTIONS_PATH}")
