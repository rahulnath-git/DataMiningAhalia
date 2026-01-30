import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import mutual_info_classif

# ==========================================
# 1. Data Loading and Exploration
# ==========================================
def load_and_explore(file_path):
    print("--- Task 1: Data Loading and Exploration ---")
    df = pd.read_csv(file_path)
    
    print(f"Number of records: {df.shape[0]}")
    print(f"Number of attributes: {df.shape[1]}")
    
    print("\nClass Distribution:")
    print(df["Result"].value_counts())
    return df

# ==========================================
# 2. Data Preprocessing
# ==========================================
def preprocess_data(df):
    print("\n--- Task 2: Data Preprocessing ---")
    
    # Handle missing values by filling numerical columns with their mean
    df.fillna(df.mean(numeric_only=True), inplace=True)
    
    # Encode target column (Result: Pass -> 1, Fail -> 0)
    le = LabelEncoder()
    df["Result"] = le.fit_transform(df["Result"])
    print(f"Encoded 'Result' mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Normalize numerical attributes using Min-Max Scaling
    scaler = MinMaxScaler()
    num_cols = ["Age", "Study_Hours", "Attendance", "Internal_Marks"]
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    print("Preprocessing completed: Data cleaned, encoded, and normalized.")
    return df

# ==========================================
# 3. Feature Selection (Information Gain)
# ==========================================
def feature_selection(df):
    print("\n--- Task 3: Feature Selection ---")
    X = df.drop("Result", axis=1)
    y = df["Result"]
    
    # Compute Information Gain using mutual_info_classif
    info_gain = mutual_info_classif(X, y, random_state=42)
    ig_results = pd.Series(info_gain, index=X.columns).sort_values(ascending=False)
    
    print("Information Gain for each attribute:")
    print(ig_results)
    
    best_feature = ig_results.idxmax()
    print(f"\nBest attribute for splitting: {best_feature}")
    return X, y

# ==========================================
# 4 & 5. Classification and Evaluation
# ==========================================
def train_and_evaluate(X, y):
    print("\n--- Task 4: Decision Tree Classification ---")
    # Split the dataset into Training (80%) and Testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Implement Decision Tree using Entropy (ID3 Concept)
    model = DecisionTreeClassifier(criterion="entropy", random_state=42)
    model.fit(X_train, y_train)
    print("Model training completed.")
    
    print("\n--- Task 5: Evaluation ---")
    y_pred = model.predict(X_test)
    
    # Compute Metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred)
    }
    
    for metric, value in metrics.items():
        print(f"{metric:10}: {value:.4f}")

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # Path to your generated dataset
    file_name = "student_performance.csv"
    
    # Run pipeline
    data = load_and_explore(file_name)
    processed_data = preprocess_data(data)
    features, target = feature_selection(processed_data)
    train_and_evaluate(features, target)