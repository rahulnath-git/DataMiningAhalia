# Data Mining Assignment 1 – Decision Tree Classification

## 📌 Problem Statement
The objective of this assignment is to build a Decision Tree classification model to predict whether a student will **Pass or Fail** based on academic and behavioral attributes.

---

## 📊 Dataset Description
A synthetic but realistic dataset containing **100 student records** was created.  
The dataset includes the following attributes:

| Attribute | Description |
|---------|-------------|
| Age | Age of the student |
| Study_Hours | Average number of study hours per day |
| Attendance | Attendance percentage |
| Internal_Marks | Internal assessment marks |
| Result | Pass / Fail (Target variable) |

The dataset is saved as `student_performance.csv`.

---

## ⚙️ Tools & Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Google Colab / VS Code

---

##  Methodology

### 1. Data Loading & Exploration
- Loaded the dataset using Pandas
- Identified:
  - Number of records
  - Number of attributes
  - Class distribution of Pass and Fail

### 2. Data Preprocessing
- Handled missing values using mean imputation
- Encoded categorical target variable (Pass/Fail → 1/0)
- Normalized numerical features using Min-Max Scaling

### 3. Feature Selection
- Used **Gini Index** to compute feature importance
- Identified the most influential attributes for classification

### 4. Decision Tree Model
- Implemented a Decision Tree classifier using Scikit-learn
- Used Gini Index as the splitting criterion
- Controlled model complexity using maximum depth

### 5. Model Evaluation
The trained model was evaluated on test data using:
- Accuracy
- Precision
- Recall
- F1-score

---

## 📈 Results
The Decision Tree model successfully classified student performance with good accuracy and balanced evaluation metrics, demonstrating the effectiveness of decision trees in rule-based academic prediction problems.

---

## 📂 Repository Structure
## 🎯 Conclusion
This project demonstrates the complete machine learning workflow, including data preprocessing, feature selection, model training, and evaluation using a Decision Tree classifier. It highlights how academic performance can be effectively predicted using data mining techniques.

---

## 👩‍💻 Author
**Riya**  
Final Year B.Tech – Computer Science Engineering