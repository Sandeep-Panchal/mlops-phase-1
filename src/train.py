# --- Import required modules ---
import mlflow
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.config import *
from src.model_registry import promote_model_if_better

# -------------------------------
# Data Preparation
# -------------------------------
def data_preparation():

    # FIX: make path consistent for local + docker + CI
    df = pd.read_csv("data/binary_class.csv")

    print(f"Shape of dataframe: {df.shape}")

    X, y = df["text"], df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


# -------------------------------
# Training Function
# -------------------------------
def train():

    # Only create experiment if not CI
    if not CI_MODE:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = data_preparation()

    pipeline = Pipeline([
        ("vectorizer", TfidfVectorizer()),
        # ("vectorizer", CountVectorizer()),  # optional
        ("clf", LogisticRegression())
    ])

    # Train
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy}")

    # -------------------------------
    # CI Mode → skip MLflow + registry
    # -------------------------------
    if CI_MODE:
        print("CI mode → skipping MLflow + registry")
        return

    # -------------------------------
    # MLflow Logging + Registry
    # -------------------------------
    with mlflow.start_run() as run:

        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(
            pipeline,
            "model",
            registered_model_name=MODEL_NAME
        )

        run_id = run.info.run_id

        print(f"Run ID: {run_id}")

        # Promotion logic
        promote_model_if_better(run_id, accuracy)


# -------------------------------
# Entry Point
# -------------------------------
if __name__ == "__main__":
    train()