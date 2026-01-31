import pandas as pd

# Load the dataset
data = pd.read_csv("students.csv")

# Basic exploration
print("Number of records:", data.shape[0])
print("Number of attributes:", data.shape[1])

print("\nClass distribution:")
print(data["Result"].value_counts())

#  DATA PREPROCESSING

# 1. Handle missing values (replace with average)
data.fillna(data.mean(numeric_only=True), inplace=True)

# 2. Convert Result (Pass/Fail) to numeric
data["Result"] = data["Result"].map({"Pass": 1, "Fail": 0})

# 3. Normalize numerical columns
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
numerical_cols = ["Age", "Study_Hours", "Attendance", "Internal_Marks"]
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

print("\nPreprocessing completed successfully!")

#    FEATURE & TARGET

X = data.drop("Result", axis=1)   # input features
y = data["Result"]                # output (Pass/Fail)

print("\nFeatures and target separated successfully!")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Data split into training and testing sets!")

#   DECISION TREE CLASSIFIER
from sklearn.tree import DecisionTreeClassifier

# Create the model (ID3 concept: use 'entropy')
model = DecisionTreeClassifier(criterion="entropy")

# Train the model
model.fit(X_train, y_train)

print("Decision Tree trained successfully!")
#    PREDICTIONS
y_pred = model.predict(X_test)

print("Predictions completed successfully!")

# -------- EVALUATION --------
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("\nModel Evaluation Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))




#  OUTPUT
"""
Number of records: 100
Number of attributes: 5

Class distribution:
Result
Fail    75
Pass    25
Name: count, dtype: int64

Preprocessing completed successfully!

Features and target separated successfully!
Data split into training and testing sets!
Decision Tree trained successfully!
Predictions completed successfully!

Model Evaluation Metrics:
Accuracy: 0.95
Precision: 1.0
Recall: 0.8333333333333334
F1 Score: 0.9090909090909091  """