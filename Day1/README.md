# 🚢 Day 1: Titanic Dataset - Data Cleaning & Preprocessing

This is my **Day 1 task** for the AI/ML Internship program.  
I worked on cleaning and preparing the Titanic dataset for machine learning.

---

## ✅ Tasks Completed

- Loaded dataset using Pandas
- Checked and handled missing values  
  - Filled missing `Age` with median  
  - Filled missing `Embarked` with most common value  
- Converted categorical columns  
  - `Sex`: male → 0, female → 1  
  - One-hot encoded `Embarked`  
- Standardized `Age` and `Fare` using StandardScaler  
- Detected and removed outliers using the IQR method  
- Created plots for data visualization:  
  - Bar chart, boxplot, histogram, pie chart, heatmap

---

## 🛠 Tools Used

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

## 📁 Folder Contents

- `main.py` – Python script with preprocessing and visualizations  
- `Figure_1.png` – Combined data visualization image  
- `Titanic-Dataset` – Contains the Titanic CSV file  
- `README.md` – This file  

---

## 📥 Dataset Source

- [Kaggle Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)  

---

## 📚 Key Learnings

- Data cleaning and preprocessing  
- Handling missing values  
- Encoding categorical features  
- Standardizing numeric data  
- Visualizing data and removing outliers
