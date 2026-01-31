import pandas as pd
import numpy as np
#creating database
np.random.seed(42)

n = 100

data = {
    "Age": np.random.randint(18, 26, n),
    "Study_Hours": np.round(np.random.uniform(1, 10, n), 1),
    "Attendance": np.random.randint(50, 101, n),
    "Internal_Marks": np.random.randint(20, 101, n)
}

df = pd.DataFrame(data)

# Rule-based target generation
df["Result"] = np.where(
    (df["Study_Hours"] >= 4) & 
    (df["Attendance"] >= 75) & 
    (df["Internal_Marks"] >= 40),
    "Pass",
    "Fail"
)

# Introduce missing values
for col in ["Study_Hours", "Attendance"]:
    df.loc[df.sample(frac=0.05).index, col] = np.nan

df.to_csv("student_dataset.csv", index=False)
print("Dataset created successfully!")

#loading data
df = pd.read_csv("student_dataset.csv")

#preprocessing 
#handling missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

#Encoding Target Variable
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["Result"] = le.fit_transform(df["Result"])  
# Pass = 1, Fail = 0

#feature normalisation
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
features = ["Age", "Study_Hours", "Attendance", "Internal_Marks"]
df[features] = scaler.fit_transform(df[features])

#feature selection
from sklearn.tree import DecisionTreeClassifier

X = df.drop("Result", axis=1)
y = df["Result"]

dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(X, y)

for feature, importance in zip(X.columns, dt.feature_importances_):
    print(f"{feature}: {importance:.4f}")

#desicion tree
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

#model training
model = DecisionTreeClassifier(criterion="entropy", max_depth=4)
model.fit(X_train, y_train)

#evaluatioin
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-score :", f1_score(y_test, y_pred))

