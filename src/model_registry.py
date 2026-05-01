import mlflow
from mlflow.tracking import MlflowClient

from src.config import *

# -------------------------------
# MLflow setup
# -------------------------------
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient()


# -------------------------------
# Get latest version safely
# -------------------------------
def get_latest_version():
    versions = client.get_latest_versions(MODEL_NAME)
    return versions[0].version if versions else None


# -------------------------------
# Get production model accuracy
# -------------------------------
def get_production_model_accuracy():
    try:
        prod_versions = client.get_latest_versions(
            MODEL_NAME,
            stages=["Production"]
        )

        if not prod_versions:
            return None

        run_id = prod_versions[0].run_id
        metrics = client.get_run(run_id).data.metrics

        return metrics.get("accuracy")

    except Exception as e:
        print(f"Error fetching production model: {e}")
        return None


# -------------------------------
# Promote model if better
# -------------------------------
def promote_model_if_better(run_id, new_acc):

    # 🔥 Get the version linked to THIS run
    versions = client.search_model_versions(f"run_id='{run_id}'")

    if not versions:
        print("No model version found for this run ❌")
        return

    version = versions[0].version

    # Move to Staging first
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage="Staging"
    )

    prod_acc = get_production_model_accuracy()

    print(f"Current Production Accuracy: {prod_acc}")
    print(f"New Model Accuracy: {new_acc}")

    # Promotion logic
    if prod_acc is None or new_acc > prod_acc:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version,
            stage="Production"
        )
        print(f"Model v{version} promoted to Production ✅")
    else:
        print(f"Model v{version} NOT promoted ❌")