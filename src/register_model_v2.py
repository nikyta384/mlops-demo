import os
import mlflow
from mlflow.tracking.client import MlflowClient

def register_model(run_id: str, model_name: str,  commit_id: str = None):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    client = MlflowClient()

    model_uri = f"runs:/{run_id}/model"
    print(f"Registering model from URI: {model_uri} as '{model_name}'")

    # Register model
    result = mlflow.register_model(model_uri, model_name)
    print(f"Model registered with version: {result.version}")
    if commit_id:
        client.set_model_version_tag(
            name=model_name,
            version=result.version,
            key="commit_id",
            value=commit_id
        )
        print(f"Set commit_id tag: {commit_id}")
    # Transition model to desired stage
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage='test',
        archive_existing_versions=True
    )
    print(f"Model version {result.version} transitioned to stage 'test'")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Register a MLflow model by run ID.")
    parser.add_argument("--run_id", required=True, help="MLflow run ID of the model to register")
    parser.add_argument("--commit", required=True, help="Commit hash or unique identifier for the model version")
    args = parser.parse_args()

    register_model(args.run_id, 'my-model', args.commit)