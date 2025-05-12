import bentoml
import argparse
import mlflow

parser = argparse.ArgumentParser()
parser.add_argument("--accuracy", required=True, type=float, help="Accuracy for current training")
parser.add_argument("--commit", required=True, help="Commit hash or unique identifier for the model version")
parser.add_argument("--mlflow_tracking_uri", required=True, help="Remote MLflow tracking URI")
parser.add_argument("--model_version", required=True, help="model_version")
args = parser.parse_args()

accuracy = args.accuracy
commit_id = args.commit
mlflow_tracking_uri = args.mlflow_tracking_uri
model_name = "my-model"
model_version = args.model_version

mlflow.set_tracking_uri(mlflow_tracking_uri)


model = mlflow.sklearn.load_model(f"models:/{model_name}/latest")
bento_model = bentoml.sklearn.save_model(
    name=f"{model_name}:{commit_id}",
    model=model,
    labels={"commit": commit_id, "version": model_version},
    metadata={"accuracy": accuracy},
)
print(f"Model registered with BentoML: {bento_model.tag}")
with open('../../model_version', 'w') as version_file:
    version_file.write(commit_id)