import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.feature_selection import mutual_info_classif

# --- Task 1: Data Loading and Exploration ---
df = pd.read_csv('student_data.csv')

print("--- Data Exploration ---")
print(f"Number of records: {df.shape[0]}")
print(f"Number of attributes: {df.shape[1]}")
print("Class distribution:\n", df['Result'].value_counts())

# --- Task 2: Data Preprocessing ---
# 1. Handle missing values (Fill with mean)
df['Study_Hours'] = df['Study_Hours'].fillna(df['Study_Hours'].mean())

# 2. Convert categorical target to numerical (Pass: 1, Fail: 0)
le = LabelEncoder()
df['Result'] = le.fit_transform(df['Result'])

# 3. Normalize numerical attributes
scaler = StandardScaler()
features = ['Age', 'Study_Hours', 'Attendance', 'Internal_Marks']
df[features] = scaler.fit_transform(df[features])

print("\nPreprocessing Complete. Missing values handled and features normalized.")

# --- Task 3: Feature Selection (Information Gain) ---
X = df[features]
y = df['Result']

importances = mutual_info_classif(X, y)
feature_importance = pd.Series(importances, index=features)

print("\n--- Feature Selection (Information Gain) ---")
print(feature_importance.sort_values(ascending=False))
print(f"Best attribute for splitting: {feature_importance.idxmax()}")

# --- Task 4: Decision Tree Classification ---
# Split data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and Train Classifier (using Gini Index)
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf.fit(X_train, y_train)

print("\nDecision Tree Model Trained.")

# --- Task 5: Evaluation ---
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n--- Evaluation Metrics ---")
print(f"Accuracy:  {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1-score:  {f1:.2f}")
print("\nDetailed Classification Report:\n", classification_report(y_test, y_pred))