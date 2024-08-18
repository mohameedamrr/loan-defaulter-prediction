# This file is for utility functions to be used in the notebook.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
plt.style.use('ggplot')
pd.set_option('display.max_columns', 200) # So we can see all columns
pd.set_option('display.max_rows',200)


def get_numerical_summary(df):
    total = df.shape[0]
    missing_columns = [col for col in df.columns if df[col].isnull().sum() > 0]
    missing_percent = {}
    for col in missing_columns:
        null_count = df[col].isnull().sum()
        per = (null_count/total) * 100
        missing_percent[col] = per
        print("{} : {} ({}%)".format(col, null_count, round(per, 3)))
    return missing_percent

def get_categorical_features(df):
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return categorical_features


def calculate_metrics(validation_data, predicted_data):
    # Calculate metrics
    accuracy = accuracy_score(validation_data, predicted_data)
    f1 = f1_score(validation_data, predicted_data)
    roc = roc_auc_score(validation_data, predicted_data)
    
    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC Score: {roc:.4f}")

def setup_model(df):
    # Split the data into features (X) and target (y)
    x = df.drop('TARGET', axis=1)
    y = df['TARGET']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    adasyn = ADASYN(sampling_strategy='minority', random_state=42)
    X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)

    X_train_res, X_val, y_train_res, y_val = train_test_split(X_train_resampled, y_train_resampled, test_size=0.3, random_state=42)
    return X_train_res,X_val,y_train_res,y_val,X_test,y_test


