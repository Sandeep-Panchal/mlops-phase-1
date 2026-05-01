import mlflow
from mlflow.tracking import MlflowClient
from config import *

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient()

# # Search for all registered models
# registered_models = client.search_registered_models()

# # Extract and print model names
# model_names = [model.name for model in registered_models]
# print("Registered Model Names:", model_names)


def get_latest_version():
    versions = client.get_latest_versions(MODEL_NAME)
    return versions[0].version if versions else None

def get_production_model_accuracy():

    try:
        prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if not prod_versions:
            return None

        run_id = prod_versions[0].run_id
        metrics = client.get_run(run_id).data.metrics
        return metrics.get("accuracy")

    except:
        return None

def promote_model_if_better(run_id, new_acc):

    # result = client.create_model_version(
    #     name=MODEL_NAME,
    #     source=f"runs:/{run_id}/model",
    #     run_id=run_id
    # )

    # version = result.version

    latest_versions = client.get_latest_versions(MODEL_NAME)[0].version

    # Move to staging
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=latest_versions,
        stage="Staging"
    )

    prod_acc = get_production_model_accuracy()

    print(f"Current Production Accuracy: {prod_acc}")
    print(f"New Model Accuracy: {new_acc}")

    # Promote logic
    if prod_acc is None or new_acc > prod_acc:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=latest_versions,
            stage="Production"
        )
        print("Promoted to Production ✅")
    else:
        print("Model not promoted ❌")