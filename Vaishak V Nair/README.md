# Student Performance Prediction using Decision Tree

This project demonstrates the complete data mining and machine learning pipeline using a **Decision Tree classifier** to predict student performance (Pass/Fail) based on academic and behavioral attributes.

## Overview
The objective of this assignment is to design a realistic dataset, preprocess it, perform feature selection, train a Decision Tree model, and evaluate its performance using standard classification metrics.

A synthetic dataset was created to simulate a real academic environment with controlled noise and missing values.

## Dataset
- Total records: 100
- Features:
  - Age
  - Study_Hours
  - Attendance
  - Internal_Marks
- Target:
  - Result (Pass / Fail)

The dataset includes missing values and non-linear relationships to ensure realism.

## Methodology
1. Dataset generation using Python
2. Data exploration and analysis
3. Handling missing values using mean imputation
4. Feature normalization
5. Feature selection using Information Gain
6. Decision Tree classification
7. Model evaluation using Accuracy, Precision, Recall, and F1-score

## Files in the Repository
- `generate_student_dataset.py`  
  Python script to generate the synthetic dataset

- `student_performance_dataset.csv`  
  Generated dataset used for training and testing

- `decision_tree_student_performance.ipynb`  
  Jupyter Notebook containing the full implementation

## Tools and Libraries
- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook

## How to Run
1. Run `generate_student_dataset.py` to generate the dataset
2. Open `decision_tree_student_performance.ipynb`
3. Execute the notebook cells sequentially

## Conclusion
The Decision Tree model demonstrates reasonable performance on a small synthetic dataset. Results highlight the impact of internal marks and attendance on student outcomes, while also showing the effect of noise and limited data size on classification performance.
