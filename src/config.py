import os

# Environment: local | docker | ci
ENV = os.getenv("ENV", "local")

# -------------------------------
# MLflow Tracking URI
# -------------------------------
if ENV == "docker":
    MLFLOW_TRACKING_URI = "http://host.docker.internal:5000"

elif ENV == "ci":
    # No MLflow server in CI → use local file store
    MLFLOW_TRACKING_URI = "file:./mlruns"

else:  # local
    MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"


# -------------------------------
# Experiment + Model Config
# -------------------------------
EXPERIMENT_NAME = "sentiment-exp"
MODEL_NAME = "SentimentModel"
THRESHOLD = 0.80


# -------------------------------
# Flags (optional but useful)
# -------------------------------
CI_MODE = ENV == "ci"
DOCKER_MODE = ENV == "docker"
LOCAL_MODE = ENV == "local"