# app/model_utils.py

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

def get_models(task_type):
    if task_type == 'classification':
        return {
            "RandomForest": RandomForestClassifier(),
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "XGBoost": XGBClassifier(verbosity=0),
            "LightGBM": LGBMClassifier(),
            "CatBoost": CatBoostClassifier(verbose=0)
        }
    else:
        return {
            "RandomForest": RandomForestRegressor(),
            "LinearRegression": LinearRegression(),
            "XGBoost": XGBRegressor(verbosity=0),
            "LightGBM": LGBMRegressor(),
            "CatBoost": CatBoostRegressor(verbose=0)
        }
