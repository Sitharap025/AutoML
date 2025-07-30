from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd

def preprocess_data(df, target_col, task_type):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Basic imputing
    X = X.copy()
    num_cols = X.select_dtypes(include='number').columns
    cat_cols = X.select_dtypes(exclude='number').columns

    if len(num_cols) > 0:
        imputer_num = SimpleImputer(strategy='mean')
        X[num_cols] = imputer_num.fit_transform(X[num_cols])

    if len(cat_cols) > 0:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        X[cat_cols] = imputer_cat.fit_transform(X[cat_cols])
        X = pd.get_dummies(X, columns=cat_cols)

    # Encode target if classification
    if task_type == 'classification':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return train_test_split(X, y, test_size=0.2, random_state=42)
