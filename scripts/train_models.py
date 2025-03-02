import os
import mlflow
import mlflow.sklearn
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from data_process import load_and_process_data
import numpy as np
import pandas as pd
from pathlib import Path

# ✅ Ensure MLflow tracking directory exists
mlflow_tracking_dir = Path("mlruns").resolve().as_uri()
mlflow.set_tracking_uri(mlflow_tracking_dir)

# ✅ Set the experiment
mlflow.set_experiment("Customer_Satisfaction_Prediction")

def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test):
    """ Train and log model using MLflow """
    with mlflow.start_run():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Log parameters & metrics
        mlflow.log_param("Model", model_name)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("F1-Score", f1)

        # ✅ Save model to MLflow
        mlflow.sklearn.log_model(model, model_name)

        print(f"✅ {model_name} - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")

    return model, accuracy  # ✅ Return model and accuracy

def save_best_model_as_pkl(model, filename="best_model.pkl"):
    """ Save the best model as a .pkl file """
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Best model saved as {filename}")

if __name__ == "__main__":
    dataset_path = "Passenger_Satisfaction.csv"

    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        exit(1)

    # Load and process data
    try:
        X_train, X_test, y_train, y_test, _, _ = load_and_process_data(dataset_path)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        exit(1)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100)
    }

    best_model = None
    best_accuracy = 0.0

    for name, model in models.items():
        trained_model, accuracy = train_and_log_model(model, name, X_train, X_test, y_train, y_test)
        
        # ✅ Save the best model as .pkl
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = trained_model

    if best_model:
        save_best_model_as_pkl(best_model)

    print("✅ Model training & logging complete.")
