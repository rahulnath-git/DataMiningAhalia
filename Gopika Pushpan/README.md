# Data Preprocessing and Decision Tree Classification using Python

## Assignment 1 – Data Mining

---

## Objective
To understand and implement the complete data mining pipeline, including data preprocessing, feature selection, classification, and evaluation using Python.

---

## Problem Statement
Create a CSV dataset with 100 realistic student records containing the following attributes:
- Age
- Study_Hours
- Attendance
- Internal_Marks
- Result (Pass / Fail)

The dataset is used to build and evaluate a Decision Tree classification model.

---

## Dataset Description
The dataset consists of **100 student records** with the following attributes:

| Attribute         | Description                                |
|------------------|--------------------------------------------|
| Age              | Age of the student                         |
| Study_Hours      | Average study hours per day                |
| Attendance       | Attendance percentage                      |
| Internal_Marks   | Internal assessment marks                  |
| Result           | Pass / Fail (Target variable)              |

The dataset includes a few missing values to demonstrate preprocessing techniques.

---

## Tasks Performed

### 1. Data Loading and Exploration
- Loaded the dataset using Python (pandas)
- Displayed:
  - Number of records
  - Number of attributes
  - Class distribution (Pass / Fail)

---

### 2. Data Preprocessing
- Handled missing values using **mean imputation**
- Normalized numerical attributes using **Min–Max normalization**
- Converted the categorical target attribute (Pass / Fail) into numerical form (1 / 0)

---

### 3. Feature Selection
- Computed **Information Gain** for all input features
- Identified the best attribute for splitting the dataset

---

### 4. Decision Tree Classification
- Implemented a Decision Tree classifier using entropy (ID3 concept)
- Split the dataset into training and testing sets
- Trained the model using the processed dataset

---

### 5. Model Evaluation
The model was evaluated using the following metrics:


---

## Tools and Technologies Used
- Python
- Google Colab
- pandas
- numpy
- scikit-learn

---

## How to Run the Project
1. Open `DataMining_Assignment1.ipynb` in Google Colab or Jupyter Notebook
2. Run all cells sequentially
3. View outputs for each task and evaluation metrics

---

## Conclusion
This assignment demonstrates the complete data mining workflow, including data preprocessing, feature selection, Decision Tree classification, and model evaluation. The results show that Decision Trees are effective for rule-based classification problems.
