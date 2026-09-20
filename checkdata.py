import pandas as pd

# Load the dataset
df = pd.read_csv(r"data\WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Show first 5 rows
print("========== FIRST 5 ROWS ==========")
print(df.head())

# Show dataset size
print("\n========== DATASET SHAPE ==========")
print(df.shape)

# Show column names
print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

# Show data types
print("\n========== DATA TYPES ==========")
print(df.dtypes)

# Check missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# Check duplicate rows
print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())

print("\n========== DATA CHECK COMPLETED ==========")