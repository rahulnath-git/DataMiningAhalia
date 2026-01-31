# ---------------------------------------------
# Data Mining Assignment 1
# Student Performance Classification
# ---------------------------------------------

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------

data = pd.read_csv("student_data.csv")

print("Total Records:", data.shape[0])
print("Total Attributes:", data.shape[1])

print("\nClass Distribution:")
print(data["Result"].value_counts())

# ---------------------------------------------
# 2. DATA PREPROCESSING
# ---------------------------------------------

# Encode target variable
label_encoder = LabelEncoder()
data["Result"] = label_encoder.fit_transform(data["Result"])

X = data.drop("Result", axis=1)
y = data["Result"]

# Feature scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print("\nData preprocessing completed")

# ---------------------------------------------
# 3. FEATURE SELECTION (GINI INDEX)
# ---------------------------------------------

gini_model = DecisionTreeClassifier(criterion="gini", random_state=21)
gini_model.fit(X_scaled, y)

print("\nFeature Importance using Gini Index:")
for feature, score in zip(X.columns, gini_model.feature_importances_):
    print(f"{feature}: {score:.4f}")

# ---------------------------------------------
# 4. MODEL TRAINING
# ---------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.3,
    random_state=21,
    stratify=y
)

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    min_samples_split=5,
    random_state=21
)

model.fit(X_train, y_train)

print("\nDecision Tree model trained successfully")

# ---------------------------------------------
# 5. MODEL EVALUATION
# ---------------------------------------------

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Fail", "Pass"]))
