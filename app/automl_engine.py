import optuna
import mlflow
import mlflow.sklearn
import json
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.base import clone
from model_util import get_models
from data_utils import preprocess_data


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
    best_model_name = None
    best_score = float("inf") if task_type == "regression" else 0
    best_model_obj = None

    mlflow.set_experiment("AutoML-App_V2")

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
                # Add more model-specific tuning here if needed

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
            results.append((model_name, round(score, 4)))

            # Update best model
            is_better = score < best_score if task_type == "regression" else score > best_score
            if is_better:
                best_model_name = model_name
                best_score = score
                best_model_obj = model_final

            # MLflow logging
            with mlflow.start_run(run_name=f"{model_name}_{task_type}"):
                mlflow.log_param("model_name", model_name)
                mlflow.log_params(best_params)
                mlflow.log_param("task_type", task_type)
                mlflow.log_metric("final_score", score)
                mlflow.set_tag("framework", "scikit-learn")
                mlflow.set_tag("automl", "optuna")

                # Save best params as artifact
                with open("best_params.json", "w") as f:
                    json.dump(best_params, f)
                mlflow.log_artifact("best_params.json")

                # Save preview of dataset
                df.head().to_csv("preview.csv", index=False)
                mlflow.log_artifact("preview.csv")

                # Log model
                mlflow.sklearn.log_model(
                    model_final,
                    artifact_path="model",
                    input_example=X_test[:5],
                    registered_model_name=None  # Optional: register model name
                )

        except Exception as e:
            print(f"⚠️ Skipping {model_name} due to error: {e}")

    #return results, best_model_name, round(best_score, 4), best_model_obj, X_test, y_test
    return results, best_model_name, best_score, best_model_obj

