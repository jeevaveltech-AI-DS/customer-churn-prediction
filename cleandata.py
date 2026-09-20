import pandas as pd

# Load the dataset
df = pd.read_csv(r"data\WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Original shape:", df.shape)

# Convert TotalCharges to numeric
# Invalid/blank values will become NaN
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Check missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Remove rows where TotalCharges is missing
df = df.dropna(subset=["TotalCharges"])

# Remove duplicate rows
df = df.drop_duplicates()

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nCleaned shape:", df.shape)

# Save cleaned dataset
df.to_csv(r"data\cleaned_churn.csv", index=False)

print("\nCleaned dataset saved successfully!")
