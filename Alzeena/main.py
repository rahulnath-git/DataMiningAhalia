import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#load the dataset
data = pd.read_csv(r"C:\Users\ALZEENA\OneDrive\Desktop\Data mining Assignment\student_dataset (1)(Sheet1).csv")
print("\n--- DATASET LOADED ---")
print(data.head())
print("Number of Records:", data.shape[0])
print("Number of Attributes:", data.shape[1])
print(" Class Distribution:")
print(data["Result"].value_counts())
# Data Preprocessing
#handling missing values
print("Missing values before handling:\n")
print(data.isnull().sum())
data.fillna(data.mean(numeric_only=True), inplace=True)
# convert categorical to numerical
le = LabelEncoder()
data["Result"] = le.fit_transform(data["Result"]) 
print(data.head())
#pass=1,fail=0
# Normalization converting numerical to the same scale
scaler = MinMaxScaler()
features = ["Age", "Study_Hours", "Attendance", "Internal_Marks"]
data[features] = scaler.fit_transform(data[features])
print("\n--- DATA AFTER NORMALIZATION ---")
print(data[features].describe())
#feature selection
#feature selection using gini importance
X = data[features]
y = data["Result"]

model = DecisionTreeClassifier()
model.fit(X, y)

importance = model.feature_importances_

print("\n--- FEATURE IMPORTANCE ---")
for f, imp in zip(features, importance):
    print(f"{f}: {imp}")
     #Best attribute
best_features = [features[i] for i in np.argsort(importance)[-2:]]
print("Selected Features for Model Training:", best_features)   
# Model Training and Evaluation
 #Train test split  
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)  
# Train model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train) 
print("X_train size:", X_train.shape)
print("X_test size:", X_test.shape)
print("y_train size:", y_train.shape)
print("y_test size:", y_test.shape)
#Evaluate model
y_pred = clf.predict(X_test)
print("\n--- MODEL EVALUATION ---")
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))
print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
print("Accuracy:", accuracy_score(y_test, y_pred)) 
 








