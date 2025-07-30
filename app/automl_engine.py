import optuna
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, mean_squared_error
from model_util import get_models
from data_utils import preprocess_data
from sklearn.base import clone

def detect_task_type(target_series):
    num_unique = target_series.nunique()
    total = len(target_series)
    if target_series.dtype == 'object' or num_unique < total * 0.05:
        return "classification"
    return "regression"

def run_automl(df, target_col, task_type):
    X_train, X_test, y_train, y_test = preprocess_data(df, target_col, task_type)
    models = get_models(task_type)

    results = []

    mlflow.set_experiment("AutoML-App")

    for model_name, model in models.items():
        try:
            def objective(trial):
                model_copy = clone(model)
                if model_name == "XGBoost":
                    model_copy.set_params(
                        max_depth=trial.suggest_int("max_depth", 3, 10),
                        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                        n_estimators=trial.suggest_int("n_estimators", 50, 200)
                    )
                elif model_name == "RandomForest":
                    model_copy.set_params(
                        n_estimators=trial.suggest_int("n_estimators", 50, 200),
                        max_depth=trial.suggest_int("max_depth", 3, 15)
                    )
                # Add more model-specific tuning logic here

                model_copy.fit(X_train, y_train)
                preds = model_copy.predict(X_test)
                return 1.0 - accuracy_score(y_test, preds) if task_type == 'classification' else mean_squared_error(y_test, preds)

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=5)

            best_params = study.best_params
            model_final = clone(model)
            model_final.set_params(**best_params)
            model_final.fit(X_train, y_train)
            preds = model_final.predict(X_test)

            score = accuracy_score(y_test, preds) if task_type == 'classification' else mean_squared_error(y_test, preds)

            results.append((model_name, score))

        except Exception as e:
            print(f"⚠️ Skipping {model_name} due to error: {e}")

    return sorted(results, key=lambda x: x[1], reverse=(task_type == "classification"))
