import pandas as pd

# -------------------------------
# STEP 1: LOAD DATASET
# -------------------------------
df = pd.read_csv("data/train.csv")

print("✅ Dataset Preview:")
print(df.head())

# -------------------------------
# STEP 2: UNDERSTAND DATA
# -------------------------------
print("\n✅ Dataset Info:")
print(df.info())

print("\n✅ Statistical Summary:")
print(df.describe())

print("\n❗ Missing Values BEFORE Cleaning:")
print(df.isnull().sum())

# -------------------------------
# STEP 3: SAVE ORIGINAL DATA
# -------------------------------
raw_df = df.copy()

# -------------------------------
# STEP 4: DATA CLEANING
# -------------------------------

# Fill categorical columns
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])

# Fill numerical columns
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())

# Important column
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

# Fix Dependents column
df['Dependents'] = df['Dependents'].replace('3+', 3)
df['Dependents'] = pd.to_numeric(df['Dependents'])

# -------------------------------
# STEP 5: CHECK AFTER CLEANING
# -------------------------------
print("\n✅ Missing Values AFTER Cleaning:")
print(df.isnull().sum())

# -------------------------------
# STEP 6: ENCODING (TEXT → NUMBERS)
# -------------------------------
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']

for col in cols:
    df[col] = le.fit_transform(df[col])

# -------------------------------
# STEP 7: DROP UNNECESSARY COLUMN
# -------------------------------
df.drop('Loan_ID', axis=1, inplace=True)

# -------------------------------
# STEP 8: SPLIT DATA
# -------------------------------
from sklearn.model_selection import train_test_split

X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# STEP 9: FEATURE SCALING
# -------------------------------
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# STEP 10: TRAIN MODEL
# -------------------------------
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

# -------------------------------
# STEP 11: PREDICTION
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# STEP 12: ACCURACY
# -------------------------------
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("\n🎯 Model Accuracy:", accuracy)

# -------------------------------
# STEP 13: BEFORE vs AFTER
# -------------------------------
print("\n🔴 BEFORE PREPROCESSING:")
print(raw_df.head())

print("\n🟢 AFTER PREPROCESSING:")
print(df.head())
# -------------------------------
# ENCODING (TEXT → NUMBERS)
# -------------------------------
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']

for col in cols:
    df[col] = le.fit_transform(df[col])

# -------------------------------
# DROP UNNECESSARY COLUMN
# -------------------------------
if 'Loan_ID' in df.columns:
    df.drop('Loan_ID', axis=1, inplace=True)

# -------------------------------
# SPLIT DATA
# -------------------------------
from sklearn.model_selection import train_test_split

X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# FEATURE SCALING
# -------------------------------
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# TRAIN MODEL
# -------------------------------
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

# -------------------------------
# PREDICTION
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# ACCURACY
# -------------------------------
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("\n🎯 Model Accuracy:", accuracy)

# -------------------------------
# BEFORE vs AFTER (FOR REPORT)
# -------------------------------
print("\n🔴 BEFORE PREPROCESSING:")
print(raw_df.head())

print("\n🟢 AFTER PREPROCESSING:")
print(df.head())
import seaborn as sns
import matplotlib.pyplot as plt

# Graph 1: Loan Status Distribution
sns.countplot(x='Loan_Status', data=raw_df)
plt.title("Loan Approval Distribution")
plt.show()
sns.boxplot(x='Loan_Status', y='ApplicantIncome', data=raw_df)
plt.title("Income vs Loan Status")
plt.savefig("income_vs_loan.png")