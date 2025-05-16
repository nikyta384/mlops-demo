# src/bentoml_register.py

import bentoml
import argparse
import mlflow
import os

parser = argparse.ArgumentParser()
parser.add_argument("--accuracy", required=True, type=float, help="Accuracy for current training")
parser.add_argument("--commit", required=True, help="Commit hash or unique identifier for the model version")
parser.add_argument("--mlflow_tracking_uri", required=True, help="Remote MLflow tracking URI")
parser.add_argument("--run_id", required=True, help="run_id")
args = parser.parse_args()

accuracy = args.accuracy
commit_id = args.commit
mlflow_tracking_uri = args.mlflow_tracking_uri
run_id = args.run_id
model_name = "my-model"

mlflow.set_tracking_uri(mlflow_tracking_uri)

# Load MLflow model by model version (run ID)
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

# Save model to BentoML with labels and metadata
bento_model = bentoml.sklearn.save_model(
    name=f"{model_name}:{commit_id}",
    model=model,
    labels={"commit": commit_id, "run_id": run_id},
    metadata={"accuracy": accuracy},
)
# Add a "latest" alias/tag pointing to this commit
bento_model.tag = f"{model_name}:latest"
bento_model._bento_model_metadata.tag.name = "latest"
bento_model.save()

print(f"Model registered with BentoML: {bento_model.tag}")

# version_filepath = os.path.join(os.path.expanduser("~"), "model_version")
# with open(version_filepath, 'w') as version_file:
#     version_file.write(commit_id)