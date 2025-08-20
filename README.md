# Healthinformatics

# Malaria prediction

This project focuses on building and deploying a machine learning model Gradient Boosting to predict malaria severity 
based on patient data, then wrapping it in a Streamlit web app so users can select details (age, sex, symptoms, etc.) 
and get predictions.

## What the Notebook Does:

Loads and explores the malaria dataset

Groups patients by age and gender

Counts how often each symptom appears

Compares symptoms between severe and non-severe malaria cases

Analyzes which age groups and genders have higher severity

Trains and evaluates machine learning models Gradient Boosting, XGBoost, Random Forest, Logistic Regression and Decision Tree

Saves the best model and scaler for later use in an app - Gradient Boosting

## Tools Used:
Python311

NumPy 1.24.4

pandas 1.5.3

scikit-learn 1.2.2

imbalanced-learn  0.11.0

XGBoost 1.7.5

matplotlib

seaborn

## How to Use:

1. Download the notebook (malaria-prediction-app.ipynb) and dataset.
2. convert the dataset to csv
3. Install the required libraries:  
   `pip install -r requirements.txt`
4. Open the notebook in Jupyter or VS Code.
5. Run each cell to see the charts and results.
6. Save the malaria_predictionn.py to your folder
7. Copy the folder path and run it in command prompt
8. Run 'streamlit run malaria_prediction.py' ---don't add the quote
9. The app will display on Localhost


## Dataset:

The notebook uses a file called `mmc1.xls`. 
