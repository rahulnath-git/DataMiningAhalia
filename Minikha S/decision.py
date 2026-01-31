import pandas as pd

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


data = pd.read_csv("student_details.csv")


print("Number of Records:", data.shape[0])
print("Number of Attributes:", data.shape[1])

print("\nClass Distribution:")
print(data['Result'].value_counts())


data.fillna(data.mean(numeric_only=True), inplace=True)

scaler = MinMaxScaler()
num_cols = ['Age', 'Study_Hours', 'Attendance', 'Internal_Marks']
data[num_cols] = scaler.fit_transform(data[num_cols])

encoder = LabelEncoder()
data['Result'] = encoder.fit_transform(data['Result'])


X = data.drop('Result', axis=1)
y = data['Result']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(criterion='entropy')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nEvaluation Metrics:")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-Score :", f1_score(y_test, y_pred))