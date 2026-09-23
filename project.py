# ============================================================
# LOAN APPROVAL PREDICTION USING RANDOM FOREST
# ============================================================

# -------------------------------
# STEP 1: IMPORT LIBRARIES
# -------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Improve visualization style
sns.set_theme(style="whitegrid")


# ============================================================
# STEP 2: LOAD DATASET
# ============================================================

df = pd.read_csv("data/train.csv")

print("=" * 60)
print("DATASET PREVIEW")
print("=" * 60)

print(df.head())


# ============================================================
# STEP 3: UNDERSTAND THE DATA
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(df.info())

print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)

print(df.describe(include="all"))

print("\n" + "=" * 60)
print("MISSING VALUES BEFORE CLEANING")
print("=" * 60)

print(df.isnull().sum())


# Save original dataset
raw_df = df.copy()


# ============================================================
# STEP 4: CHECK DUPLICATE RECORDS
# ============================================================

print("\n" + "=" * 60)
print("DUPLICATE RECORDS")
print("=" * 60)

duplicates = df.duplicated().sum()

print("Number of duplicate rows:", duplicates)

if duplicates > 0:
    df = df.drop_duplicates()
    print("Duplicate rows removed.")

else:
    print("No duplicate rows found.")


# ============================================================
# STEP 5: VISUALIZE MISSING VALUES
# ============================================================

plt.figure(figsize=(10, 5))

sns.heatmap(
    df.isnull(),
    cbar=False,
    cmap="viridis"
)

plt.title("Missing Values Before Preprocessing")
plt.xlabel("Features")
plt.ylabel("Rows")

plt.tight_layout()
plt.show()


# ============================================================
# STEP 6: DATA CLEANING
# ============================================================

# -------------------------------
# Categorical columns
# -------------------------------

categorical_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Self_Employed"
]

for col in categorical_columns:

    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])


# -------------------------------
# Numerical columns
# -------------------------------

numerical_columns = [
    "LoanAmount",
    "Loan_Amount_Term"
]

for col in numerical_columns:

    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())


# -------------------------------
# Credit History
# -------------------------------

df["Credit_History"] = df["Credit_History"].fillna(
    df["Credit_History"].mode()[0]
)


# ============================================================
# STEP 7: FIX DEPENDENTS COLUMN
# ============================================================

df["Dependents"] = df["Dependents"].replace("3+", "3")

df["Dependents"] = pd.to_numeric(
    df["Dependents"],
    errors="coerce"
)

df["Dependents"] = df["Dependents"].fillna(
    df["Dependents"].median()
)


# ============================================================
# STEP 8: CHECK MISSING VALUES AFTER CLEANING
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES AFTER CLEANING")
print("=" * 60)

print(df.isnull().sum())


# ============================================================
# STEP 9: LOAN STATUS DISTRIBUTION
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    x="Loan_Status",
    data=raw_df
)

plt.title("Loan Approval Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Number of Applicants")

plt.tight_layout()
plt.show()


# ============================================================
# STEP 10: INCOME VS LOAN STATUS
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="Loan_Status",
    y="ApplicantIncome",
    data=raw_df
)

plt.title("Applicant Income vs Loan Status")
plt.xlabel("Loan Status")
plt.ylabel("Applicant Income")

plt.tight_layout()

plt.savefig(
    "income_vs_loan.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# STEP 11: CO-APPLICANT INCOME VS LOAN STATUS
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="Loan_Status",
    y="CoapplicantIncome",
    data=raw_df
)

plt.title("Coapplicant Income vs Loan Status")
plt.xlabel("Loan Status")
plt.ylabel("Coapplicant Income")

plt.tight_layout()
plt.show()


# ============================================================
# STEP 12: LOAN AMOUNT DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="LoanAmount",
    kde=True,
    bins=30
)

plt.title("Loan Amount Distribution")
plt.xlabel("Loan Amount")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# ============================================================
# STEP 13: CORRELATION HEATMAP
# ============================================================

numeric_df = df.select_dtypes(
    include=np.number
)

plt.figure(figsize=(10, 7))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Feature Correlation Heatmap")

plt.tight_layout()
plt.show()


# ============================================================
# STEP 14: PREPARE FEATURES AND TARGET
# ============================================================

# Remove Loan_ID because it does not contain useful
# predictive information.

if "Loan_ID" in df.columns:
    df = df.drop("Loan_ID", axis=1)


# Convert target:
# Y = 1
# N = 0

df["Loan_Status"] = df["Loan_Status"].map({
    "Y": 1,
    "N": 0
})


X = df.drop(
    "Loan_Status",
    axis=1
)

y = df["Loan_Status"]


# ============================================================
# STEP 15: IDENTIFY CATEGORICAL AND NUMERICAL FEATURES
# ============================================================

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


print("\n" + "=" * 60)
print("CATEGORICAL FEATURES")
print("=" * 60)

print(categorical_features)


print("\n" + "=" * 60)
print("NUMERICAL FEATURES")
print("=" * 60)

print(numerical_features)


# ============================================================
# STEP 16: TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================================
# STEP 17: PREPROCESSING PIPELINE
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        # Numerical features
        (
            "num",
            StandardScaler(),
            numerical_features
        ),

        # Categorical features
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)


# ============================================================
# STEP 18: RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(

    n_estimators=300,

    max_depth=10,

    min_samples_split=5,

    min_samples_leaf=2,

    random_state=42,

    class_weight="balanced"
)


# ============================================================
# STEP 19: CREATE COMPLETE PIPELINE
# ============================================================

pipeline = Pipeline(

    steps=[

        (
            "preprocessing",
            preprocessor
        ),

        (
            "model",
            model
        )
    ]
)


# ============================================================
# STEP 20: TRAIN MODEL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST MODEL")
print("=" * 60)

pipeline.fit(
    X_train,
    y_train
)

print("Training completed successfully.")


# ============================================================
# STEP 21: MAKE PREDICTIONS
# ============================================================

y_pred = pipeline.predict(X_test)


# ============================================================
# STEP 22: MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# ============================================================
# STEP 23: CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Rejected",
            "Approved"
        ],
        zero_division=0
    )
)


# ============================================================
# STEP 24: CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Rejected",
        "Approved"
    ],
    yticklabels=[
        "Rejected",
        "Approved"
    ]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# STEP 25: FEATURE IMPORTANCE
# ============================================================

# Get trained Random Forest
trained_model = pipeline.named_steps["model"]

# Get transformed feature names
feature_names = pipeline.named_steps[
    "preprocessing"
].get_feature_names_out()

importances = trained_model.feature_importances_

feature_importance_df = pd.DataFrame({

    "Feature": feature_names,

    "Importance": importances

}).sort_values(
    by="Importance",
    ascending=False
)


print("\n" + "=" * 60)
print("TOP 15 IMPORTANT FEATURES")
print("=" * 60)

print(
    feature_importance_df.head(15)
)


# ============================================================
# STEP 26: FEATURE IMPORTANCE GRAPH
# ============================================================

top_features = feature_importance_df.head(10)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
    "feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# STEP 27: BEFORE VS AFTER PREPROCESSING
# ============================================================

print("\n" + "=" * 60)
print("BEFORE PREPROCESSING")
print("=" * 60)

print(raw_df.head())


print("\n" + "=" * 60)
print("AFTER BASIC CLEANING")
print("=" * 60)

print(df.head())


# ============================================================
# STEP 28: EXAMPLE PREDICTION
# ============================================================

sample = X_test.iloc[[0]]

prediction = pipeline.predict(
    sample
)

probability = pipeline.predict_proba(
    sample
)[0][1]


print("\n" + "=" * 60)
print("EXAMPLE LOAN PREDICTION")
print("=" * 60)

if prediction[0] == 1:

    print("Loan Prediction: APPROVED")

else:

    print("Loan Prediction: REJECTED")


print(
    f"Approval Probability: {probability:.2%}"
)


# ============================================================
# STEP 29: FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PROJECT SUMMARY")
print("=" * 60)

print(f"Dataset Size      : {raw_df.shape}")
print(f"Training Samples  : {X_train.shape[0]}")
print(f"Testing Samples   : {X_test.shape[0]}")
print(f"Model             : Random Forest Classifier")
print(f"Accuracy          : {accuracy:.2%}")
print(f"Precision         : {precision:.2%}")
print(f"Recall            : {recall:.2%}")
print(f"F1 Score          : {f1:.2%}")

print("\n✅ Loan prediction model completed successfully.")
