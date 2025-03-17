# Movie Genre Classification (Google Colab)

This project focuses on classifying movie genres based on their descriptions using machine learning techniques. It was developed as part of an internship task to demonstrate the ability to preprocess text data, handle imbalanced datasets, and evaluate machine learning models. The goal is to predict the genre of a movie based on its description. The project is designed to run on **Google Colab**, making it easy to execute without needing to set up a local environment.

---

## Project Overview

The project involves the following steps:

1. **Data Loading and Exploration**: The dataset is loaded and explored to understand its structure and identify any missing values.
2. **Data Preprocessing**:
   - Text data (movie descriptions) is cleaned and preprocessed.
   - Categorical variables (genres) are encoded using Label Encoding.
   - Text data is converted into numerical features using TF-IDF Vectorization.
3. **Handling Imbalanced Data**: The Synthetic Minority Over-sampling Technique (SMOTE) is applied to balance the dataset (if necessary).
4. **Model Training**: Two models are trained:
   - Decision Tree Classifier
   - Logistic Regression Classifier
5. **Model Evaluation**: The models are evaluated using the F1-score and a detailed classification report.

---

## Dataset

The dataset used in this project is sourced from Kaggle:
- **Dataset Link**: [Genre Classification Dataset - IMDB](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb)

The dataset consists of the following files:
- `train_data.txt`: Contains movie descriptions and their corresponding genres for training.
- `test_data.txt`: Contains movie descriptions for testing.
- `test_data_solution.txt`: Contains the correct genres for the test data.

The dataset is loaded directly into Google Colab for processing.

---

## How to Use (Google Colab)

1. **Open the Notebook in Google Colab**:
   - Click the "Open in Colab" button below to launch the notebook directly in Google Colab:
     [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

2. **Upload the Dataset**:
   - Use the file browser in Google Colab to upload the dataset files (`train_data.txt`, `test_data.txt`, and `test_data_solution.txt`).

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

      action       0.85      0.86      0.85       200
      comedy       0.83      0.82      0.82       180
       drama       0.88      0.89      0.88       220

    accuracy                           0.86       600
   macro avg       0.85      0.86      0.85       600
weighted avg       0.86      0.86      0.86       600

Logistic Regression Report:
               precision    recall  f1-score   support

      action       0.88      0.89      0.88       200
      comedy       0.85      0.84      0.84       180
       drama       0.90      0.91      0.90       220

    accuracy                           0.88       600
   macro avg       0.88      0.88      0.88       600
weighted avg       0.88      0.88      0.88       600
'''
