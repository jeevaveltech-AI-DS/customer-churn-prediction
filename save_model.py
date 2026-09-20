import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression


# Load cleaned dataset
df = pd.read_csv(r"data\cleaned_churn.csv")


# Convert Yes/No columns into numbers
encoder = LabelEncoder()

df["gender"] = encoder.fit_transform(df["gender"])
df["Partner"] = encoder.fit_transform(df["Partner"])
df["Dependents"] = encoder.fit_transform(df["Dependents"])
df["PhoneService"] = encoder.fit_transform(df["PhoneService"])
df["PaperlessBilling"] = encoder.fit_transform(df["PaperlessBilling"])
df["Churn"] = encoder.fit_transform(df["Churn"])


# Convert categorical columns into numerical columns
df = pd.get_dummies(
    df,
    columns=[
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod"
    ],
    drop_first=True
)


# Remove customer ID
df = df.drop("customerID", axis=1)


# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create Logistic Regression model
model = LogisticRegression(max_iter=1000)


# Train model
model.fit(X_train, y_train)


# Save model
joblib.dump(model, "churn_model.pkl")


# Save feature names
joblib.dump(X.columns.tolist(), "model_features.pkl")


print("Model saved successfully!")
print("churn_model.pkl created")
print("model_features.pkl created")