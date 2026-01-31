import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import MinMaxScaler
import sys

# Redirect output to file
sys.stdout = open('output.txt', 'w')

# Step 1: Load the original dataset
print("Loading original dataset...")
df = pd.read_csv('dataset.csv')
print(f"Original dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Class distribution: {df['Result'].value_counts().to_dict()}")

# Step 2: Data Preprocessing
print("\nPreprocessing data...")

# Handle missing values (if any)
numerical_cols = ['Age', 'Study Hours', 'Attendance', 'Internal Marks']
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mean(), inplace=True)

if df['Result'].isnull().sum() > 0:
    df['Result'].fillna(df['Result'].mode()[0], inplace=True)

# Normalize numerical attributes
scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Convert categorical to numerical
df['Result'] = df['Result'].map({'Fail': 0, 'Pass': 1})

print("Preprocessing completed.")
print(f"Preprocessed dataset shape: {df.shape}")

# Step 3: Train Decision Tree Model
print("\nTraining Decision Tree Classifier...")

X = df.drop('Result', axis=1)
y = df['Result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train, y_train)

print("Model trained successfully.")

# Step 4: Evaluate on Test Data
print("\nEvaluating on test data...")

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nEvaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

print("\nFeature Importances:")
for feature, importance in zip(X.columns, clf.feature_importances_):
    print(f"{feature}: {importance:.4f}")

# Close the output file
sys.stdout.close()