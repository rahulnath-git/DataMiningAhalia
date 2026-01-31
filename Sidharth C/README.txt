# Decision Tree Classifier Project Summary

## Overview
This project implements a decision tree classifier using scikit-learn to predict student pass/fail outcomes based on academic features. The model achieves perfect accuracy on the test set.

## Dataset
- **File**: `dataset.csv`
- **Size**: 100 rows, 5 columns
- **Features**: Age, Study Hours, Attendance, Internal Marks
- **Target**: Result (Pass/Fail)
- **Distribution**: 54 Pass, 46 Fail

## Steps
1. **Data Loading**: Read CSV and display basic info.
2. **Preprocessing**: Handle missing values, normalize numerical features (MinMaxScaler), encode target (Fail=0, Pass=1).
3. **Training**: Split data (80% train, 20% test), train DecisionTreeClassifier with entropy criterion.
4. **Evaluation**: Compute accuracy, precision, recall, F1-score, and feature importances.

## Results
- **Accuracy**: 1.0000
- **Precision**: 1.0000
- **Recall**: 1.0000
- **F1 Score**: 1.0000
- **Key Features**: Attendance (61.6%), Internal Marks (38.4%)

## Files
- `main.py`: Main script
- `dataset.csv`: Original data
- `output.txt`: Execution results
- `preprocessed_dataset.csv`: Processed data (optional)

## How to Run
Place all files in the same directory. Run `main.py` to generate `output.txt`.