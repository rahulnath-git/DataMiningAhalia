# CST466 – Data Mining  
## Assignments & Tutorials (KTU – B.Tech CSE)

This repository contains **Assignments and Tutorials** designed for the **CST466 – Data Mining** course under **APJ Abdul Kalam Technological University (KTU)**.  
The content aligns with **course outcomes (COs)**, **internal assessment standards**, and **practical exposure using Python**.

---

##  Course Information

- **Course Code:** CST466  
- **Course Name:** Data Mining  
- **Programme:** B.Tech Computer Science & Engineering  
- **University:** APJ Abdul Kalam Technological University (KTU)

---

#  ASSIGNMENTS

---

## 🔹 ASSIGNMENT 1  
### **Title:** Data Preprocessing and Decision Tree Classification using Python

###  Objective
To understand and implement the **complete data mining pipeline**, including preprocessing, feature selection, classification, and evaluation.

- **COs Mapped:** CO1, CO2, CO3  
- **Type:** Individual Assignment  
- **Submission Date:** **31 January 2025**  
- **Mode:** Online  
- **GitHub Repository:**  
   `rahulnath-git/DataMiningAssignment1`

---

###  Problem Statement

Create a **CSV dataset with 100 realistic records** containing the following attributes:

| Attribute | Description |
|---------|-------------|
| Age | Age of the student |
| Study_Hours | Average study hours per day |
| Attendance | Attendance percentage |
| Internal_Marks | Internal assessment marks |
| Result | Pass / Fail (Target class) |

---

###  Tasks

#### 1. Data Loading and Exploration
- Load the dataset using Python
- Display:
  - Number of records
  - Number of attributes
  - Class distribution

#### 2. Data Preprocessing
- Handle missing values using suitable methods
- Normalize numerical attributes
- Convert categorical attributes (if any) to numerical form

#### 3. Feature Selection
- Compute **Information Gain** or **Gini Index**
- Identify the best attribute for splitting

#### 4. Decision Tree Classification
- Implement a Decision Tree classifier  
  *(ID3 concept or library-based implementation)*
- Train the model using the processed dataset

#### 5. Evaluation
- Predict results on test data
- Compute:
  - Accuracy
  - Precision
  - Recall
  - F1-score

---

###  Expected Results
- Cleaned and preprocessed dataset
- Trained Decision Tree model
- Clearly printed evaluation metrics

---

### 🛠 Guidelines
- Use any programming language *(Python preferred)*  
- Code must be well-documented  
- Upload source code and dataset to GitHub

---

##  ASSIGNMENT 2  
### **Title:** Clustering, Association Rule Mining and Text Analysis

###  Objective
To apply **advanced data mining techniques** such as:
- Clustering
- Association Rule Mining
- Text Mining

- **COs Mapped:** CO3, CO4, CO5  
- **Type:** Group Assignment  
- **Submission Date:** **30 March 2025**  
- **Mode:** Online  
- **GitHub Repository:**  
   `rahulnath-git/DataMiningAssignment2`

---

###  Given Datasets
1. Transaction dataset (customer purchases)
2. Text dataset (short customer reviews)

---

###  Part A: Clustering

1. Load a dataset with numerical attributes  
   *(e.g., age, income, spending score)*
2. Apply:
   - PAM (K-Medoids) **or**
   - DBSCAN clustering
3. Display:
   - Number of clusters
   - Cluster size
4. Visualize clusters using **2D plots**

---

### Part B: Association Rule Mining

1. Use the transaction dataset
2. Implement:
   - Apriori **or**
   - FP-Growth algorithm
3. Generate:
   - Frequent itemsets
   - Strong association rules
4. Display rules with:
   - Support
   - Confidence

---

### Part C: Text Mining

1. Preprocess text data:
   - Tokenization
   - Stop-word removal
2. Create a **term-frequency matrix**
3. Implement a **keyword-based search**
4. Compute:
   - Precision
   - Recall

---

###  Expected Outcomes
- Clustered data with visualization
- Frequent itemsets and association rules
- Processed text data with retrieval metrics

---

#  TUTORIALS

---

##  TUTORIAL 1  
### **Module 1: Introduction to Data Mining & Data Warehousing**

1. Explain differences between operational databases and data warehouses with examples.
2. Draw and explain a star schema for a sales data warehouse.
3. Describe OLAP operations. How does roll-up differ from drill-down?
4. Explain the complete KDD process with a neat diagram.
5. Discuss data mining functionalities with applications.

---

##  TUTORIAL 2  
### **Module 2: Data Preprocessing**

1. Why is data preprocessing necessary? Explain three data quality issues.
2. Explain methods to handle missing values with examples.
3. Differentiate between normalization and standardization.
4. Explain attribute subset selection with techniques.
5. Apply equal-width and equal-frequency discretization  
   *(Dataset: 5, 8, 10, 15, 18, 20, 25, 30; 4 bins)*

---

##  TUTORIAL 3  
### **Module 3: Classification & Clustering**

1. Explain decision tree construction principle and entropy.
2. Explain Information Gain calculation for binary classification.
3. Compare ID3 and SLIQ algorithms.
4. Explain DBSCAN and noise handling.
5. Distinguish between partitioning and hierarchical clustering.

---

##  TUTORIAL 4  
### **Module 4: Association Rule Mining**

1. Define support, confidence, and frequent itemsets.
2. Explain Apriori property and pruning.
3. Describe the Partition algorithm.
4. Explain Pincer Search and its advantages.
5. Compare Apriori and FP-Growth.

---

##  TUTORIAL 5  
### **Module 5: Advanced Data Mining Techniques**

1. Differentiate web content, structure, and usage mining.
2. Explain PageRank with an example.
3. Explain HITS algorithm and hub/authority concept.
4. Explain text preprocessing steps.

---
