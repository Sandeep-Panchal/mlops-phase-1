import mlflow
import mlflow.pyfunc

from src.config import *

# -------------------------------
# MLflow setup
# -------------------------------
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_URI = f"models:/{MODEL_NAME}/Production"

# Lazy loaded model
model = None

def load_model():
    global model

    if model is None:
        print("Loading model from MLflow...")
        model = mlflow.pyfunc.load_model(MODEL_URI)


def predict(text: str):
    load_model()
    sentiment = int(model.predict([text])[0])
    return "Positive" if sentiment == 1 else "Negative"


# -------------------------------
# Test run
# -------------------------------
if __name__ == "__main__":
    print(predict("you are a bad boy"))