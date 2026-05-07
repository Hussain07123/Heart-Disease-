"""
This script generates a sample heart disease dataset for demonstration.
In production, replace with the real Cleveland Heart Disease dataset from:
https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci
"""
import pandas as pd
import numpy as np

np.random.seed(42)
n = 303

data = {
    "age": np.random.randint(29, 77, n),
    "sex": np.random.randint(0, 2, n),
    "cp": np.random.randint(0, 4, n),
    "trestbps": np.random.randint(94, 200, n),
    "chol": np.random.randint(126, 564, n),
    "fbs": np.random.randint(0, 2, n),
    "restecg": np.random.randint(0, 3, n),
    "thalach": np.random.randint(71, 202, n),
    "exang": np.random.randint(0, 2, n),
    "oldpeak": np.round(np.random.uniform(0, 6.2, n), 1),
    "slope": np.random.randint(0, 3, n),
    "ca": np.random.randint(0, 4, n),
    "thal": np.random.randint(0, 3, n),
}

# Simple rule-based target for demo purposes
df = pd.DataFrame(data)
df["target"] = (
    (df["age"] > 55).astype(int) +
    (df["cp"] > 1).astype(int) +
    (df["thalach"] < 140).astype(int) +
    (df["exang"] == 1).astype(int) +
    (df["oldpeak"] > 2).astype(int)
) >= 3
df["target"] = df["target"].astype(int)

df.to_csv("heart.csv", index=False)
print(f"Dataset saved: {len(df)} rows, {df['target'].sum()} positive cases")
