import mlflow.pyfunc
from src.config import *

mlflow.set_tracking_uri(DOCKER_MLFLOW_TRACKING_URI)

MODEL_URI = f"models:/{MODEL_NAME}/Production"
model = mlflow.pyfunc.load_model(MODEL_URI)


def predict(text: str):
    sentiment = int(model.predict([text])[0])
    return "Positive" if sentiment == 1 else "Negative"

if __name__=="__main__":

    text = "you are a bad boy"
    sentiment = predict(text)
    print(sentiment)