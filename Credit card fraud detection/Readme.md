# Credit Card Fraud Detection (Google Colab)

This project focuses on detecting fraudulent credit card transactions using machine learning techniques. It was developed as part of an **internship task** to demonstrate the ability to preprocess data, handle imbalanced datasets, and evaluate machine learning models. The goal is to identify potentially fraudulent activities based on transaction data. The project is designed to run on **Google Colab**, making it easy to execute without needing to set up a local environment.

## Project Overview

The project involves the following steps:

1. **Data Loading and Exploration**: The dataset is loaded and explored to understand its structure and identify any missing values.
2. **Data Preprocessing**: 
   - Unnecessary columns are dropped.
   - Categorical variables are encoded using Label Encoding.
   - Date and time features are extracted.
   - Numeric features are scaled using StandardScaler.
3. **Handling Imbalanced Data**: The Synthetic Minority Over-sampling Technique (SMOTE) is applied to balance the dataset.
4. **Model Training**: Two models are trained:
   - Decision Tree Classifier
   - Random Forest Classifier
5. **Model Evaluation**: The models are evaluated using the F1-score and a detailed classification report.

## Dataset

The dataset used in this project is `fraudTrain.csv`, which contains transaction details along with a binary label indicating whether the transaction is fraudulent or not. You can download the dataset from the following link:

- **Dataset Link**: [fraudTrain.csv](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

The dataset is loaded directly into Google Colab for processing.

---

## How to Use (Google Colab)

1. **Open the Notebook in Google Colab**:
   - Click the "Open in Colab" button below to launch the notebook directly in Google Colab:
     [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vzQi05FBd22f0BwItS5bNK8FYx7FISAG)

2. **Upload the Dataset**:
   - The notebook assumes the dataset (`fraudTrain.csv`) is uploaded to Google Colab. You can upload it manually or use the provided code to load it from a URL.

3. **Run the Notebook**:
   - Execute each cell in the notebook sequentially to load the data, preprocess it, train the models, and evaluate their performance.

---

## Dependencies

The following Python libraries are required to run the notebook:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imblearn

These libraries are pre-installed in Google Colab, so no additional installation is required.

---

## Results

The performance of the models is evaluated using the F1-score and detailed classification reports. Below are the results from the evaluation:

### Decision Tree Classifier Report:
```plaintext
Decision Tree Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00    128916
           1       0.00      0.00      0.00       214

    accuracy                           1.00    129130
   macro avg       0.50      0.50      0.50    129130
weighted avg       1.00      1.00      1.00    129130



Random Forest Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00    128916
           1       0.00      0.00      0.00       214

    accuracy                           1.00    129130
   macro avg       0.50      0.50      0.50    129130
weighted avg       1.00      1.00      1.00    129130'''
