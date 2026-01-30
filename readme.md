# Student Performance Prediction using Decision Tree

## Project Overview
This project involves building a machine learning pipeline to predict whether a student will **Pass** or **Fail** based on academic and demographic features. The project utilizes a synthetic dataset containing 100 student records with added noise to simulate real-world data challenges like missing values and non-linear relationships.

## Dataset Description
The dataset `student_performance.csv` contains the following attributes:
* **Age**: Age of the student (18–25).
* **Study_Hours**: Average hours spent studying per day.
* **Attendance**: Percentage of classes attended.
* **Internal_Marks**: Marks obtained in internal assessments.
* **Result**: Target class (Pass/Fail).

---

## Implementation Methodology

### 1. Data Loading and Exploration
* Loaded the dataset using `pandas`.
* Analyzed the class distribution to understand the balance between passing and failing students.

### 2. Data Preprocessing
* **Handling Missing Values**: Implemented mean imputation for numerical columns.
* **Label Encoding**: Converted categorical target `Result` into numerical format (Fail = 0, Pass = 1).
* **Normalization**: Applied `MinMaxScaler` to numerical features to scale them to a range of [0, 1].

### 3. Feature Selection
* Calculated **Information Gain** using the Mutual Information criterion to identify which attributes contribute most to the prediction.

### 4. Classification
* Split the data into 80% training and 20% testing sets.
* Trained a **Decision Tree Classifier** using the **Entropy** (ID3 concept) criterion.

---

## Results

### Execution Output
```text
--- Task 1: Data Loading and Exploration ---
Number of records: 100
Number of attributes: 5

Class Distribution:
Result
Fail    71
Pass    29
Name: count, dtype: int64

--- Task 2: Data Preprocessing ---
Encoded 'Result' mapping: {'Fail': 0, 'Pass': 1}
Preprocessing completed: Data cleaned, encoded, and normalized.

--- Task 3: Feature Selection ---
Information Gain for each attribute:
Attendance        0.204204
Internal_Marks    0.101677
Study_Hours       0.076092
Age               0.000000
dtype: float64

Best attribute for splitting: Attendance

--- Task 4: Decision Tree Classification ---
Model training completed.

--- Task 5: Evaluation ---
Accuracy  : 0.8500
Precision : 0.6667
Recall    : 1.0000
F1-score  : 0.8000