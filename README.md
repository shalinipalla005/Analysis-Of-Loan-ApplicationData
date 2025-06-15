# Analysis of Loan Application Data
# Loan Approval Prediction

This project is a machine learning pipeline for predicting loan approval status based on applicant data. It uses Python, pandas, scikit-learn, imbalanced-learn, and visualization libraries to preprocess data, train multiple classifiers, tune hyperparameters, and make predictions on new user input.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Setup & Usage](#setup--usage)
- [Project Structure](#project-structure)
- [Modeling Approach](#modeling-approach)
- [Results](#results)
- [How to Predict for New Applicants](#how-to-predict-for-new-applicants)
- [References](#references)

---

## Overview

This project aims to automate the process of loan approval by building a predictive model using historical loan application data. The workflow includes:

- Data cleaning and preprocessing
- Exploratory data analysis (EDA)
- Handling class imbalance
- Feature encoding and scaling
- Training and evaluating multiple classifiers (Logistic Regression, Random Forest, SVM)
- Hyperparameter tuning with GridSearchCV
- Feature importance analysis
- Interactive prediction for new applicants

---

## Features

- **Data Cleaning:** Handles missing values and encodes categorical variables.
- **EDA:** Visualizes distributions and relationships in the data.
- **Imbalance Handling:** Uses oversampling to balance approved/denied classes.
- **Model Comparison:** Compares Logistic Regression, Random Forest, and SVM.
- **Hyperparameter Tuning:** Optimizes Random Forest with GridSearchCV.
- **Feature Importance:** Visualizes which features matter most.
- **User Prediction:** Interactive CLI for predicting loan eligibility for new applicants.

---

## Dataset

- The dataset should be named `data.csv` and placed in the project root.
- It must include columns such as:  
  `Loan_ID`, `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`, `Property_Area`, `Loan_Status`.

---

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn

Install dependencies with:

```sh
pip install -r requirements.txt
```

**Example `requirements.txt`:**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
```

---

## Setup & Usage

1. **Clone the repository and navigate to the project directory.**

2. **Place your `data.csv` file in the root directory.**

3. **Run the notebook:**

   - Open `main.ipynb` in Jupyter Notebook or VS Code.
   - Run all cells sequentially.

4. **For new predictions:**
   - At the end of the notebook, follow the CLI prompts to enter applicant details and get a loan eligibility prediction.

---

## Project Structure

```
.
├── main.ipynb        # Main Jupyter notebook with all code
├── data.csv             # Input dataset (not included)
└── README.md            # Project documentation
```

---

## Modeling Approach

1. **Data Loading & Inspection:**  
   Loads the CSV, checks for missing values, and inspects data types.

2. **EDA:**  
   - Visualizes class balance and feature distributions.
   - Uses count plots and pie charts for categorical variables.

3. **Data Cleaning:**  
   - Drops `Loan_ID`.
   - Fills missing values (mean/mode as appropriate).
   - Removes rows with missing `Credit_History`.

4. **Encoding:**  
   - Label encodes categorical variables.

5. **Feature Correlation:**  
   - Visualizes correlations with a heatmap.

6. **Balancing:**  
   - Uses `RandomOverSampler` to balance the classes.

7. **Splitting & Scaling:**  
   - Splits into train/test sets.
   - Scales features with `StandardScaler`.

8. **Model Training & Evaluation:**  
   - Trains Logistic Regression, Random Forest, and SVM.
   - Evaluates with accuracy, precision, recall, F1, ROC AUC, and cross-validation.
   - Plots confusion matrices.

9. **Model Comparison:**  
   - Plots a bar chart comparing model accuracies.

10. **Hyperparameter Tuning:**  
    - Uses `GridSearchCV` to optimize Random Forest.

11. **Feature Importance:**  
    - Plots feature importances for the best Random Forest model.

12. **User Prediction:**  
    - Collects user input via CLI.
    - Encodes and scales input.
    - Predicts eligibility and probability.

---

## Results

- **Best Model:** Random Forest (with tuned hyperparameters)
- **Metrics:**  
  - Accuracy, Precision, Recall, F1, ROC AUC (see notebook output for details)
- **Feature Importance:**  
  - Visualized in the notebook; shows which applicant features most influence approval.

---

## How to Predict for New Applicants

At the end of the notebook, you will be prompted to enter applicant details such as gender, marital status, dependents, education, employment, income, loan amount, term, credit history, and property area. The model will output:

- **Eligibility:** Whether the applicant is eligible for a loan.
- **Probability:** The model's confidence in the prediction.

---

## References

- [scikit-learn documentation](https://scikit-learn.org/)
- [imbalanced-learn documentation](https://imbalanced-learn.org/)
- [pandas documentation](https://pandas.pydata.org/)
- [seaborn documentation](https://seaborn.pydata.org/)

---
