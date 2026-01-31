"""
Data Mining Assignment 1
Data Preprocessing and Decision Tree Classification
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. DATASET CREATION (100 RECORDS)

np.random.seed(7)   # different seed → originality

students = {
    "Age": np.random.randint(18, 24, 100),
    "Study_Hours": np.round(np.random.uniform(1, 6, 100), 1),
    "Attendance": np.random.randint(55, 100, 100),
    "Internal_Marks": np.random.randint(25, 50, 100)
}

df = pd.DataFrame(students)

# Target variable logic
df["Result"] = np.where(
    (df["Study_Hours"] >= 2.5) &
    (df["Attendance"] >= 70) &
    (df["Internal_Marks"] >= 35),
    "Pass", "Fail"
)

# Save dataset
df.to_csv("student_data.csv", index=False)

# 2. DATA LOADING & EXPLORATION

df = pd.read_csv("student_data.csv")

print("Number of records:", df.shape[0])
print("Number of attributes:", df.shape[1])

print("\nClass Distribution:")
print(df["Result"].value_counts())

# 3. DATA PREPROCESSING

# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Encode target
df["Result"] = df["Result"].map({"Pass": 1, "Fail": 0})

X = df.drop("Result", axis=1)
y = df["Result"]

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print("\nData preprocessing completed.")

# 4. FEATURE SELECTION (GINI INDEX)

feature_model = DecisionTreeClassifier(criterion="gini", random_state=7)
feature_model.fit(X_scaled, y)

print("\nFeature Importance using Gini Index:")
for feature, importance in zip(X.columns, feature_model.feature_importances_):
    print(f"{feature}: {round(importance, 3)}")

# 5. DECISION TREE CLASSIFICATION

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=7
)

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=7
)

model.fit(X_train, y_train)

print("\nDecision Tree model trained.")

# 6. MODEL EVALUATION

y_pred = model.predict(X_test)

print("\nModel Evaluation Metrics:")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-score :", f1_score(y_test, y_pred))
