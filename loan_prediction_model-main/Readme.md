# Loan Prediction Model

## Overview
This project implements a machine learning model to predict loan approval outcomes based on applicant data. The model uses various classification algorithms to determine whether a loan application should be approved or rejected.

## Dataset
The dataset used in this project is `loan_data.csv`, containing the following features:

- **person_age**: Age of the applicant  
- **person_gender**: Gender of the applicant  
- **person_education**: Education level of the applicant  
- **person_income**: Income of the applicant  
- **person_emp_exp**: Employment experience of the applicant  
- **person_home_ownership**: Home ownership status  
- **loan_amnt**: Loan amount requested  
- **loan_intent**: Purpose of the loan  
- **loan_int_rate**: Interest rate of the loan  
- **loan_percent_income**: Loan amount as a percentage of income  
- **cb_person_cred_hist_length**: Credit history length  
- **credit_score**: Credit score of the applicant  
- **previous_loan_defaults_on_file**: Whether the applicant has previous loan defaults  
- **loan_status**: Target variable (0 = not approved, 1 = approved)  

## Data Preprocessing

1. **Handling Missing Values**: Checked for null values; none were found.  
2. **Data Formatting**: Verified data types and ensured categorical variables were properly encoded.  
3. **Encoding Categorical Variables**:  
   - Label encoding for `person_education`  
   - One-hot encoding for `person_home_ownership`, `previous_loan_defaults_on_file`, `person_gender`, and `loan_intent`  
4. **Feature Scaling**: Standardized numerical features using `StandardScaler`.  

## Model Training and Evaluation
The following machine learning models were implemented and evaluated:

| Model                     | Accuracy  |
|----------------------------|-----------|
| Logistic Regression        | 89.22%    |
| Decision Tree Classifier   | 89.9%     |
| Random Forest Classifier   | 93.17%    |
| XGBoost Classifier         | 93.66%    |

### Model Performance
The XGBoost model was selected as the final model due to its superior performance:

- **Accuracy**: 93.66%  
- **Confusion Matrix**:  

