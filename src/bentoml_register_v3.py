import bentoml
import argparse
import mlflow
import os
import subprocess
from bentoml import Bento

parser = argparse.ArgumentParser()
parser.add_argument("--accuracy", required=True, type=float, help="Accuracy for current training")
parser.add_argument("--commit", required=True, help="Commit hash or unique identifier for the model version")
parser.add_argument("--mlflow_tracking_uri", required=True, help="Remote MLflow tracking URI")
parser.add_argument("--run_id", required=True, help="run_id")
parser.add_argument("--service_file", default="service.py", help="Path to BentoML service definition file")
parser.add_argument("--docker_image_tag", default=None, help="Tag for the Docker image (defaults to Bento tag)")
args = parser.parse_args()

accuracy = args.accuracy
commit_id = args.commit
mlflow_tracking_uri = args.mlflow_tracking_uri
run_id = args.run_id
model_name = "my-model"
service_file = args.service_file
docker_image_tag = args.docker_image_tag

# Set MLflow tracking URI
mlflow.set_tracking_uri(mlflow_tracking_uri)

# Load MLflow model by run ID
model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

# Save model to BentoML with labels and metadata
bento_model = bentoml.sklearn.save_model(
    name=f"{model_name}:{commit_id}",
    model=model,
    labels={"commit": commit_id, "run_id": run_id},
    metadata={"accuracy": accuracy},
)

print(f"Model registered with BentoML: {bento_model.tag}")

# Write commit ID to version file
version_filepath = os.path.join(os.path.expanduser("~"), "model_version")
with open(version_filepath, 'w') as version_file:
    version_file.write(commit_id)

# Create Bento by building the service with the registered model
bento = bentoml.build(
    # name=f"{model_name}",
    # tag=f"{model_name}:{commit_id}",
    service="service.py",
    version=commit_id,
    include=["*.py", "src/*.py"],
    labels={
        "commit_id": commit_id,
    },
    python=dict(
        packages=["scikit-learn", "pydantic"],
    ),
)

print(f"Bento created: {bento.tag}")
