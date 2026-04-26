import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv("customer_churn.csv")

# =========================
# 2. Data Cleaning
# =========================
# Drop unnecessary column
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

# Fill missing values
df.fillna(0, inplace=True)

# Convert target column
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# =========================
# 3. Feature Engineering
# =========================
# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Split features & target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# =========================
# 4. Model (Pipeline)
# =========================
pipeline = Pipeline([
    ("scaler", StandardScaler()),              # Fix convergence issue
    ("model", LogisticRegression(max_iter=2000))
])

# Train model
pipeline.fit(X, y)

# =========================
# 5. Save Model & Columns
# =========================
pickle.dump(pipeline, open("model.pkl", "wb"))
pickle.dump(X.columns, open("columns.pkl", "wb"))

print("✅ Model trained and saved successfully!")
