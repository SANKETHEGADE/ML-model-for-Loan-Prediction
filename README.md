# Loan Prediction ML Model

An end-to-end machine learning project that predicts whether a loan application will be approved or rejected, with a Flask web app for real-time predictions.

## Overview

Banks receive thousands of loan applications and manually reviewing each one is slow and inconsistent. This project automates that decision using a Random Forest classifier trained on applicant data, and wraps it in a simple web interface where anyone can enter their details and get an instant result.

## Pipeline 

1. **Data exploration** (`project.py`) — loads the dataset, inspects structure, checks missing values, and visualizes loan approval distribution and income vs. approval trends.
2. **Preprocessing & training** (`train_model.py`) — cleans the data, encodes categorical features, scales numerical inputs, trains a Random Forest classifier, and saves the model and scaler to disk.
3. **Web app** (`app.py`) — a Flask app that loads the saved model, takes applicant details from a form, scales the input, and returns a real-time approval prediction.

## Features used
 
| Feature | Type | Preprocessing |
|---|---|---| 
| Gender | Categorical | Label Encoding |
| Married | Categorical | Label Encoding |
| Dependents | Mixed (includes "3+") | Cleaned → Numeric |
| Education | Categorical | Label Encoding |
| Self Employed | Categorical | Label Encoding |
| Applicant Income | Numerical | Standard Scaling |
| Loan Amount | Numerical | Mean imputation + Standard Scaling |
| Loan Amount Term | Numerical | Mean imputation + Standard Scaling |
| Credit History | Binary | Mode imputation |
| Property Area | Categorical | Label Encoding |

## Tech stack

- Python, scikit-learn (Random Forest, StandardScaler, LabelEncoder)
- pandas, NumPy
- Flask
- joblib (model serialization)
- seaborn, matplotlib (EDA visualizations)

## Project structure

| File | Purpose |
|---|---|
| `project.py` | Full EDA + preprocessing + training + visualization pipeline |
| `train_model.py` | Clean training script — preprocesses data, trains model, saves `model.pkl` and `scaler.pkl` |
| `app.py` | Flask web app — takes form input, scales it, returns loan approval prediction |
| `data/train.csv` | Training dataset | 
| `model.pkl` | Saved Random Forest model |
| `scaler.pkl` | Saved StandardScaler |
| `templates/index.html` | Web interface |

## Results

- Model: Random Forest Classifier
- Test accuracy: ~76% (80/20 train-test split)
- Output: "Loan Approved " or "Loan Not Approved " with confidence

## How to run

```bash
pip install flask scikit-learn pandas numpy joblib seaborn matplotlib

# Step 1: Train the model and save it
python train_model.py

# Step 2: Launch the web app
python app.py
```

Then open `http://localhost:5000`, fill in the applicant details, and click Predict.
