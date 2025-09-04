# Loan Prediction Model

## Overview
This project implements a machine learning model to predict loan approval outcomes based on applicant data. It uses various classification algorithms to determine whether a loan application should be approved or rejected.

## Dataset
The dataset used is `loan_data.csv` with the following features:

- **person_age**: Age of the applicant  
- **person_gender**: Gender of the applicant  
- **person_education**: Education level of the applicant  
- **person_income**: Income of the applicant  
- **person_emp_exp**: Employment experience  
- **person_home_ownership**: Home ownership status  
- **loan_amnt**: Loan amount requested  
- **loan_intent**: Purpose of the loan  
- **loan_int_rate**: Interest rate of the loan  
- **loan_percent_income**: Loan amount as a percentage of income  
- **cb_person_cred_hist_length**: Credit history length  
- **credit_score**: Credit score of the applicant  
- **previous_loan_defaults_on_file**: Previous loan defaults  
- **loan_status**: Target variable (0 = not approved, 1 = approved)

## Data Preprocessing
1. **Handling Missing Values**: No missing values in the dataset.  
2. **Encoding Categorical Variables**:  
   - Label encoding for `person_education`  
   - One-hot encoding for `person_home_ownership`, `previous_loan_defaults_on_file`, `person_gender`, and `loan_intent`  
3. **Feature Scaling**: Standardized numerical features using `StandardScaler`.

## Model Training and Evaluation
| Model                     | Accuracy  |
|----------------------------|-----------|
| Logistic Regression        | 89.22%    |
| Decision Tree Classifier   | 89.9%     |
| Random Forest Classifier   | 93.17%    |
| XGBoost Classifier         | 93.66%    |

The XGBoost model was selected as the final model due to its superior performance.

### XGBoost Performance
- **Accuracy**: 93.66%  
- **Confusion Matrix**:  
- **Classification Report**:  
  - Class 0 – Precision: 0.95, Recall: 0.97, F1-score: 0.96  
  - Class 1 – Precision: 0.89, Recall: 0.81, F1-score: 0.85  

## Feature Importance
![Feature Importance](feature_importance.png)

## ROC Curve
![ROC Curve](roc_curve.png)

## Installation
1. Clone the repository:  
```bash
git clone "https://github.com/Aman-pr/Project-Machine-learning-/new/main/loan_prediction_model-main"
Install dependencies:

pip install pandas scikit-learn xgboost matplotlib


Run the Jupyter notebook:

jupyter notebook Loanprediction_model.ipynb

Model Serialization

The trained XGBoost model is saved as xgboost_model.pkl using pickle for future use.

Usage Example
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained model
with open('xgboost_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Prepare new data (apply same preprocessing steps as in the notebook)
# ...

# Make predictions
predictions = model.predict(new_data)

Project Structure
loan-prediction-model/
├── Loanprediction_model.ipynb
├── loan_data.csv
├── xgboost_model.pkl
├── feature_importance.png
├── roc_curve.png
└── README.md

Conclusion

The XGBoost model demonstrated the best performance with 93.66% accuracy. It can assist financial institutions in making informed loan approval decisions, reducing risk, and improving efficiency.

Future Improvements

Hyperparameter tuning for further optimization

Exploration of additional ensemble methods

Implementation of deep learning approaches

Development of a web-based interface for real-time predictions

Integration with real-time data sources for more accurate predictions

License
