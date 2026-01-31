# Import libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -------------------------------
# STEP 1: Load Dataset
# -------------------------------

data = pd.read_csv("student_data.csv")

print("Dataset Preview:")
print(data.head())

print("\nNumber of Records:", data.shape[0])
print("Number of Attributes:", data.shape[1])

print("\nClass Distribution:")
print(data['Result'].value_counts())

# -------------------------------
# STEP 2: Data Preprocessing
# -------------------------------

# Handle missing values
data.fillna(data.mean(numeric_only=True), inplace=True)

# Encode Pass/Fail to numbers
encoder = LabelEncoder()
data['Result'] = encoder.fit_transform(data['Result'])

# Normalize numerical attributes
scaler = MinMaxScaler()

columns = ['Age', 'Study_Hours', 'Attendance', 'Internal_Marks']
data[columns] = scaler.fit_transform(data[columns])

print("\nPreprocessing Completed")

# -------------------------------
# STEP 3: Feature Selection
# -------------------------------

X = data.drop('Result', axis=1)
y = data['Result']

temp_model = DecisionTreeClassifier(criterion='gini')
temp_model.fit(X, y)

print("\nFeature Importance (Gini Index):")

for col, score in zip(X.columns, temp_model.feature_importances_):
    print(col, ":", score)

# -------------------------------
# STEP 4: Train Decision Tree
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

print("\nDecision Tree Model Trained Successfully")

# -------------------------------
# STEP 5: Evaluation
# -------------------------------

y_pred = dt_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Evaluation Results:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
