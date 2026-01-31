# Assignment 1: Data Preprocessing and Decision Tree Classification
# Author: Upanya
# Course: Data Mining & Warehousing

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ----------------------- 1. Create Dataset -----------------------
np.random.seed(42)
data = {
    'Age': np.random.randint(18, 25, size=100),
    'Study_Hours': np.random.uniform(1, 8, size=100).round(2),
    'Attendance': np.random.uniform(60, 100, size=100).round(2),
    'Internal_Marks': np.random.randint(20, 50, size=100)
}

data['Result'] = np.where(
    (data['Study_Hours'] > 4) & (data['Attendance'] > 75) & (data['Internal_Marks'] > 30),
    'Pass', 'Fail'
)

df = pd.DataFrame(data)

# Introduce some missing values
df.loc[np.random.choice(df.index, 5, replace=False), 'Study_Hours'] = np.nan
df.to_csv("student_data.csv", index=False)

print("Sample Data:")
print(df.head())

# ----------------------- 2. Data Loading and Exploration -----------------------
df = pd.read_csv("student_data.csv")
print("\nNumber of records:", df.shape[0])
print("Number of attributes:", df.shape[1])
print("\nClass distribution:\n", df['Result'].value_counts())

# ----------------------- 3. Data Preprocessing -----------------------
df['Study_Hours'].fillna(df['Study_Hours'].mean(), inplace=True)

scaler = MinMaxScaler()
df[['Age', 'Study_Hours', 'Attendance', 'Internal_Marks']] = scaler.fit_transform(
    df[['Age', 'Study_Hours', 'Attendance', 'Internal_Marks']]
)

le = LabelEncoder()
df['Result'] = le.fit_transform(df['Result'])

print("\nAfter Preprocessing:")
print(df.head())

# ----------------------- 4. Feature Selection -----------------------
X = df.drop('Result', axis=1)
y = df['Result']

model = DecisionTreeClassifier(criterion='entropy')
model.fit(X, y)

feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importance:\n", feature_importance)

# ----------------------- 5. Decision Tree Classification -----------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# ----------------------- 6. Evaluation -----------------------
print("\nModel Evaluation:")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 2))
print("Precision:", round(precision_score(y_test, y_pred), 2))
print("Recall:", round(recall_score(y_test, y_pred), 2))
print("F1-Score:", round(f1_score(y_test, y_pred), 2))

print("\nExecution Completed Successfully!")
