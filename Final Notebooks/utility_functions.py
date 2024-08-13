# This file is for utility functions to be used in the notebook.
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

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

def calculate_metrics(validation_data, predicted_data):
    accuracy = accuracy_score(validation_data,predicted_data)
    f1_score = f1_score(validation_data,predicted_data)
    roc = roc_auc_score(validation_data,predicted_data)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"ROC-AUC Score: {roc:.4f}")
