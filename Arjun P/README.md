# Data-Mining-Assignment
Data Mining Assignment of Arjun P (ATP22CS017) S8 CSE

# Student Performance Prediction using Decision Trees

## Overview
This project demonstrates a machine learning pipeline to predict student performance (Pass/Fail) based on various academic and demographic factors. It involves data loading, exploration, preprocessing, feature selection, model training using a Decision Tree Classifier, and evaluation of the model's performance.

## Dataset
The dataset used is `Students_Dataset.csv`, which contains information about student attributes and their final result. The dataset is derived from `Math-Students Performance Data` from https://www.kaggle.com/datasets/adilshamim8/math-students created by Adil Shamim. The original dataset was edited accordingly to meet the requirements of the assignment.
- **Number of Records:** 100
- **Number of Attributes:** 5
- **Features:** `Age`, `Study_Hours`, `Attendance`, `Internal_Marks`
- **Target Variable:** `Result` (Pass/Fail)

### Data Exploration
- Numerical attributes' statistics:
  - `Age`: Mean 16.68, Std 1.32, Range 15-20
  - `Study_Hours`: Mean 2.04, Std 0.87, Range 1-4
  - `Attendance`: Mean 94.73, Std 6.60, Range 62-100
  - `Internal_Marks`: Mean 21.62, Std 6.72, Range 8-37
- The `Result` class distribution is also visualized.

## Methodology

### 1. Data Preprocessing
- **Missing Values:** No missing values were found in the dataset.
- **Categorical to Numerical Conversion:** The 'Result' column was mapped from {'Pass': 1, 'Fail': 0}.
- **Train-Test Split:** The dataset was split into training (80%) and testing (20%) sets with `random_state=42`.
- **Normalization:** Numerical features were scaled using `MinMaxScaler`.

### 2. Feature Selection
- **Mutual Information:** Mutual information was calculated to identify the most important features.
- **Key Finding:** `Internal_Marks` was identified as the most important feature with an importance score of `0.2266`.

### 3. Model Training
- **Model:** A Decision Tree Classifier was used.
- **Parameters:** `criterion='entropy'`, `random_state=42`.
- The trained decision tree is visualized to understand its decision-making process.

### 4. Evaluation
The model was evaluated on the test set using standard classification metrics:
- **Accuracy:** 0.90
- **Precision:** 0.93
- **Recall:** 0.93
- **F1-score:** 0.93

## How to Run
1. Ensure you have a Google Colab environment or a Python environment with the necessary libraries installed.
2. Upload `Students_Dataset.csv` to `/content/drive/MyDrive/Colab Notebooks/Data Mining Assignment /` or modify the `pd.read_csv` path accordingly.
3. Run all cells in the Jupyter/Colab notebook sequentially.

## Libraries Used
- `pandas`
- `matplotlib`
- `sklearn`

## Author
- Arjun P
- Roll Number: ATP22CS017
- Class: S8 CSE

