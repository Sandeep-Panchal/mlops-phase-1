# --- Import required modules ---
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from config import *
from model_registry import promote_model_if_better

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# --- Prepare training data ---
def data_preparation():

    # Load the sentiment analysis dataset
    df = pd.read_csv("../data/binary_class.csv")

    print(f"Head of the dataframe {df.head()}")
    print(f"Shape of the dataframe: {df.shape}")
    print(f"Columns of the dataframe: {df.columns}")

    X, y = df["text"], df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    print(f"Shapes of x and y train: {X_train.shape}, {y_train.shape}")
    print(f"Shapes of x and y test: {X_test.shape}, {y_test.shape}")

    return X_train, X_test, y_train, y_test

# --- Train the model and log in mlflow ---
def train():

    X_train, X_test, y_train, y_test = data_preparation()

    with mlflow.start_run() as run:

        pipeline = Pipeline([
            ("vectorizer", TfidfVectorizer()),
            # ("vectorizer", CountVectorizer()),
            ("clf", LogisticRegression())
        ])

        # Train the model
        pipeline.fit(X_train, y_train)

        test_predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, test_predictions)

        mlflow.log_metric("accuracy", accuracy)

        # log + register
        mlflow.sklearn.log_model(
            pipeline,
            "model",
            registered_model_name=MODEL_NAME
        )

        run_id = run.info.run_id

        print(f"Accuracy: {accuracy}")
        print(f"Run ID: {run_id}")

        promote_model_if_better(run_id, accuracy)


if __name__ == "__main__":
    train()


