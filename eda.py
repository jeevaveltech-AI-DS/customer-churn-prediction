import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv(r"data\cleaned_churn.csv")

# ==============================
# 1. Churn Distribution
# ==============================

plt.figure(figsize=(6, 4))

sns.countplot(data=df, x="Churn")

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.show()


# ==============================
# 2. Churn vs Tenure
# ==============================

plt.figure(figsize=(8, 5))

sns.boxplot(data=df, x="Churn", y="tenure")

plt.title("Customer Churn vs Tenure")
plt.xlabel("Churn")
plt.ylabel("Tenure (Months)")

plt.show()


# ==============================
# 3. Churn vs Contract
# ==============================

plt.figure(figsize=(8, 5))

sns.countplot(data=df, x="Contract", hue="Churn")

plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")

plt.xticks(rotation=15)

plt.show()


# ==============================
# 4. Churn vs Monthly Charges
# ==============================

plt.figure(figsize=(8, 5))

sns.boxplot(data=df, x="Churn", y="MonthlyCharges")

plt.title("Customer Churn vs Monthly Charges")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")

plt.show()


print("EDA completed successfully!")