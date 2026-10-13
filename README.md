# CreditWise ML

CreditWise ML is a small end-to-end loan approval prediction project built on the provided `loan_approval_data.csv` dataset. It trains a machine learning model to estimate whether a loan is likely to be approved and exposes that prediction in a Streamlit app.

## Project goals

- Clean and prepare a real-world loan dataset
- Train a robust binary classification model
- Track validation metrics such as accuracy, precision, recall, and AUC
- Provide a simple UI for making approval predictions on new applicant data
- Generate predictions for rows that do not yet have an approval label

## Dataset

The project uses a synthetic loan approval dataset with fields including:

- applicant income and co-applicant income
- employment status, age, marital status, education level, and gender
- existing loans, debt-to-income ratio, savings, collateral value
- loan amount, loan term, purpose, property area, and employer category
- target column: `Loan_Approved` (`Yes` / `No`)

## Models and evaluation

The project compares these three permitted classification algorithms on the same reproducible 80/20 stratified hold-out test set:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes

All three models use the same leakage-safe preprocessing pipeline:

- median imputation and standardization for numeric features
- most-frequent imputation and one-hot encoding for categorical features
- preprocessing fitted only on the training split

`train_model.py` records accuracy, precision, recall, F1-score, confusion matrix, and ROC AUC for every model in `model_metrics.json`. The application model is selected using the highest ROC AUC because the UI displays predicted probabilities; F1-score and accuracy are deterministic tie-breakers. If the scores are tied, Logistic Regression is used as the first-listed baseline.

## Repository structure

```text
creditwise-ml/
├── app.py                     # Streamlit app
├── train_model.py            # Model training and prediction export
├── loan_approval_data.csv    # Loan approval dataset
├── creditwise_selected_model.joblib # Selected permitted pipeline (created at runtime)
├── model_metrics.json        # Comparison metrics (created at runtime)
├── requirements.txt          # Python dependencies
├── .gitignore                # Ignore generated files
├── .streamlit/
│   └── config.toml
└── README.md
```

## Setup

```bash
cd creditwise-ml
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (typically `http://localhost:8501`).

## App features

The Streamlit app supports two prediction modes:

- Single applicant review: enter demographic and financial data from a form and get a probability-based approval decision
- CSV batch upload: upload a file with the same applicant columns and receive a decision for each record
- Model-signal chart: inspect compatible model coefficients or class-mean separation for a single prediction

The output is an educational estimate and must not be treated as a real financial lending decision.

## Train the model manually

```bash
python train_model.py
```

This script:

1. loads the loan dataset
2. drops records without a label from training
3. trains the pipeline
4. compares the three permitted models and selects the deployment model using the documented ROC AUC criterion
5. saves the selected model to `creditwise_selected_model.joblib`
6. saves the comparison metrics to `model_metrics.json`
7. writes predicted approval scores for unlabeled records to `predictions.csv`

## Example prediction output

The app uses the same preprocessing pipeline as training to predict approval probability for a new applicant and displays a Yes/No decision along with the recommendation confidence.

## License

This project is provided as a sample ML application for educational and demonstration purposes.
