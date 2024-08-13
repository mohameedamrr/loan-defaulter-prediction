import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import ADASYN
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
plt.style.use('ggplot')
pd.set_option('display.max_columns', 200) # So we can see all columns
pd.set_option('display.max_rows',200)

df = pd.read_csv('D:/Banque Misr Internship/Loan Datasets/sampled_data.csv')
df.head(5)


df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].abs()
df['DAYS_BIRTH'] = df['DAYS_BIRTH'].abs()
df.head()

categorical_columns = [
    'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 
    'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 
    'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 
    'ORGANIZATION_TYPE'
]
# One-hot encode the categorical columns
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Check the resulting DataFrame
df_encoded.head()
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

get_numerical_summary(df_encoded)
df_cleaned = df_encoded.dropna()
# Split the data into features (X) and target (y)
x = df_cleaned.drop('TARGET', axis=1)
y = df_cleaned['TARGET']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
adasyn = ADASYN(sampling_strategy='minority', n_neighbors=5, random_state=42)
X_res, y_res = adasyn.fit_resample(X_train, y_train)


# Split the resampled data into training and validation sets
X_train_res, X_val_res, y_train_res, y_val_res = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# Standardize the training data
scaler = StandardScaler()
X_train_res_scaled = scaler.fit_transform(X_train_res)

# Standardize the validation data using the same scaler
X_val_res_scaled = scaler.transform(X_val_res)

# Initialize the Logistic Regression model
logreg = LogisticRegression(C=0.1,max_iter=5000, random_state=42)
#Saga handles large datasets and can be used for L1 (lasso) regularization.
# Train the model on the resampled training data
logreg.fit(X_train_res_scaled, y_train_res)

# Predict on the validation set
y_pred_logreg = logreg.predict(X_val_res_scaled)
y_pred_proba_logreg = logreg.predict_proba(X_val_res_scaled)[:, 1]

# Calculate F1 and ROC-AUC
f1 = f1_score(y_val_res, y_pred_logreg)
roc_auc = roc_auc_score(y_val_res, y_pred_proba_logreg)
# Evaluate the model
print("Logistic Regression Model")
print("Accuracy:", accuracy_score(y_val_res, y_pred_logreg))
print(f1)
print(roc_auc)

from sklearn.model_selection import cross_val_score

cv_f1_scores = cross_val_score(logreg, X_train_res_scaled, y_train_res, cv=5, scoring='f1')
cv_roc_auc_scores = cross_val_score(logreg, X_train_res_scaled, y_train_res, cv=5, scoring='roc_auc')

print("Cross-Validated F1 Score:", cv_f1_scores.mean())
print("Cross-Validated ROC-AUC Score:", cv_roc_auc_scores.mean())