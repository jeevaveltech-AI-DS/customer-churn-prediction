import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


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


# Convert categorical columns into numbers
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


# Create Decision Tree
model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)


# Train model
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)


# Evaluate model
print("Decision Tree Results")

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
