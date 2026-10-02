# Customer Churn Prediction

Machine learning project developed as part of my CodSoft Machine Learning Internship.

## Project Overview

This project predicts whether a customer is likely to churn based on customer demographics, services, and account information.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Logistic Regression
- StandardScaler
- Joblib
- Streamlit

## Machine Learning Approach

1. Loaded the customer churn dataset.
2. Converted TotalCharges to numeric values.
3. Removed unnecessary customer ID information.
4. Converted categorical variables using one-hot encoding.
5. Split the data into training and testing sets.
6. Scaled numerical features.
7. Trained a Logistic Regression model with balanced class weights.
8. Evaluated the model using accuracy, precision, recall, F1-score, and ROC-AUC.
9. Saved the trained model, scaler, and feature columns.
10. Created a Streamlit application for customer churn prediction.

## Results

- Dataset: 7,043 customers
- Test set: 1,407 customers
- Accuracy: 72.56%
- Churn recall: 79%
- Churn precision: 49%
- Churn F1-score: 61%

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt