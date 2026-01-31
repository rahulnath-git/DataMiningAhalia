import pandas as pd

df = pd.read_csv("student_data.csv")

print("Number of records:", df.shape[0])
print("Number of attributes:", df.shape[1])

print("\nClass Distribution:")
print(df["Result"].value_counts())

df.fillna(df.mean(numeric_only=True), inplace=True)

print("\nMissing values handled")

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["Result"] = le.fit_transform(df["Result"])

print("\nResult column encoded")

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

features = ["Age", "Study_Hours", "Attendance", "Internal_Marks"]
df[features] = scaler.fit_transform(df[features])

print("\nData normalized")
from sklearn.feature_selection import mutual_info_classif

X = df[features]
y = df["Result"]

info_gain = mutual_info_classif(X, y)

print("\nInformation Gain:")
for feature, ig in zip(features, info_gain):
    print(feature, ":", round(ig, 4))

best_feature = features[info_gain.argmax()]
print("\nBest attribute for splitting:", best_feature)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(criterion="entropy")
model.fit(X_train, y_train)

print("\nDecision Tree model trained")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_pred = model.predict(X_test)

print("\nEvaluation Metrics:")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-score :", f1_score(y_test, y_pred))

