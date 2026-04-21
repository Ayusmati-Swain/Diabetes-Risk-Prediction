# Predicting Diabetes with Machine Learning: An Applied Classification Approach

## Overview
This project applies machine learning techniques to predict diabetes status using health, demographic, and lifestyle variables. The work follows a complete applied ML pipeline, including data cleaning, exploratory data analysis (EDA), feature preparation, model training, evaluation, and final model comparison.

The objective is to classify individuals into three categories based on **HbA1c thresholds**:

- **Normal**
- **Prediabetes**
- **Diabetes**

## Problem Statement
Diabetes is one of the most common chronic diseases worldwide, and early detection is important for reducing long-term complications. This project uses a diabetes prediction dataset containing medical, demographic, and lifestyle information to build machine learning models that can classify diabetes risk accurately.

## Dataset
Source: Kaggle Diabetes Prediction Dataset

The dataset contains **10,000 records** and **20 original features**, including:

- Age
- Sex
- Ethnicity
- BMI
- Waist Circumference
- Fasting Blood Glucose
- HbA1c
- Blood Pressure
- Cholesterol measures
- GGT
- Serum Urate
- Physical Activity Level
- Dietary Intake Calories
- Alcohol Consumption
- Smoking Status
- Family History of Diabetes
- Previous Gestational Diabetes

## Project Workflow

### 1. Data Import and Inspection
The dataset was loaded using Pandas and inspected for:

- data types
- missing values
- summary statistics
- dataset shape

### 2. Data Cleaning
The main missing values were found in the `Alcohol_Consumption` column. These were handled by replacing null values with the mode of the column.

### 3. Target Variable Creation
A new multiclass target variable called `HbA1c_Category` was created using HbA1c thresholds:

- **Normal:** HbA1c ≤ 5.7
- **Prediabetes:** 5.7 < HbA1c ≤ 6.4
- **Diabetes:** HbA1c > 6.4

### 4. Exploratory Data Analysis (EDA)
EDA was performed to understand how health indicators vary across the three HbA1c categories.

#### Key observations:
- **Fasting blood glucose** increases strongly with diabetes severity
- **BMI** and **waist circumference** rise across Normal → Prediabetes → Diabetes
- **Age** shows an increasing pattern across categories
- **Blood pressure** and **cholesterol measures** also tend to be higher in diabetic groups
- Most numeric variables had **weak pairwise correlations**, suggesting low multicollinearity

### 5. Feature Engineering
Categorical variables were converted into dummy variables using one-hot encoding.

### 6. Train-Test Split
The data was split into:
- **70% training set**
- **30% test set**

Stratified sampling was used to preserve class distribution.

### 7. Model Building
Two classification algorithms were trained and evaluated:

- **Logistic Regression**
- **Random Forest Classifier**

Logistic Regression was trained on scaled features using `StandardScaler`, while Random Forest was trained on the original encoded features.

## Model Performance

### Logistic Regression
**Test Accuracy:** 99.47%

#### Classification Summary
- Excellent performance across all three classes
- Very strong precision, recall, and F1-score
- Slight misclassification for the **Prediabetes** class

#### ROC-AUC
- Near-perfect multiclass ROC-AUC performance

### Random Forest
**Test Accuracy:** 100%

#### Classification Summary
- Perfect classification on the test set
- Precision, Recall, and F1-score of **1.00** for all classes

#### ROC-AUC
- Perfect multiclass ROC-AUC performance

## Final Model Comparison

| Model | Accuracy | F1-Score | ROC-AUC |
|------|----------|----------|---------|
| Logistic Regression | 0.9947 | 0.9832 | 0.9998 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 |

## Final Recommendation
Based on overall performance, **Random Forest** was selected as the final model because it outperformed Logistic Regression on all evaluation metrics.

It was better able to capture complex and non-linear relationships in the dataset, making it the most accurate model for this classification task.

## Feature Importance
According to the Random Forest model, the most important predictors included:

- Fasting Blood Glucose
- Cholesterol HDL
- Cholesterol Total
- Cholesterol LDL
- Waist Circumference
- GGT

These features contributed most strongly to classifying diabetes risk.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Project Structure
```text
project-folder/
│── Data/
│   └── diabetes_dataset.csv
│── Code.ipynb
│── README.md
```
##How to Run the Project
1. Clone the repository.
```
git clone <your-repo-link>
```
2. Open the project folder in VS Code or Jupyter Notebook.
3. Install required libraries.
```
pip install pandas numpy matplotlib seaborn scikit-learn
```
4. Run the notebook "Code.ipynb"
