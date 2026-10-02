import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Remove rows with missing TotalCharges
df = df.dropna(subset=["TotalCharges"])

# Remove customer ID because it is not useful for prediction
df = df.drop("customerID", axis=1)

# Convert target variable
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

# Convert categorical columns into numerical columns
df = pd.get_dummies(
    df,
    drop_first=True
)

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale numerical columns
scaler = StandardScaler()

numerical_columns = ["tenure", "MonthlyCharges", "TotalCharges"]

X_train[numerical_columns] = scaler.fit_transform(
    X_train[numerical_columns]
)

X_test[numerical_columns] = scaler.transform(
    X_test[numerical_columns]
)

# Train Logistic Regression model
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train, y_train)
# Save model, scaler, and feature columns
joblib.dump(model, "churn_model.pkl")
joblib.dump(scaler, "churn_scaler.pkl")
joblib.dump(X.columns.tolist(), "churn_features.pkl")

print("\nModel, scaler, and features saved successfully!")
# Predictions
y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nROC-AUC Score:", roc_auc_score(y_test, y_probability))