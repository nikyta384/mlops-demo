import bentoml
import argparse
import mlflow
import os

parser = argparse.ArgumentParser()
parser.add_argument("--accuracy", required=True, type=float, help="Accuracy for current training")
parser.add_argument("--commit", required=True, help="Commit hash or unique identifier for the model version")
parser.add_argument("--mlflow_tracking_uri", required=True, help="Remote MLflow tracking URI")
parser.add_argument("--model_file", required=False, default="model.pkl", help="Local model file path")
args = parser.parse_args()

accuracy = args.accuracy
commit_id = args.commit
mlflow_tracking_uri = args.mlflow_tracking_uri
model_file = args.model_file
model_name = "my-model"

mlflow.set_tracking_uri(mlflow_tracking_uri)

if accuracy > 0.85:
    try:
        with mlflow.start_run() as run:
            # Log model properly using mlflow.sklearn.log_model
            # Load your local sklearn model:
            import joblib
            sklearn_model = joblib.load(model_file)

            mlflow.sklearn.log_model(sklearn_model, artifact_path="model")

            # Register from this run artifact path
            model_uri = f"runs:/{run.info.run_id}/model"
            mv = mlflow.register_model(model_uri, model_name)

            # Add commit tag to model version
            client = mlflow.tracking.MlflowClient()
            client.set_model_version_tag(model_name, mv.version, "commit", commit_id)
            with open('model_version', 'w') as version_file:
                version_file.write(commit_id)

            print(f"Model registered to MLflow: {model_name}, version: {mv.version}")

    except Exception as e:
        print(f"Error logging and registering model: {e}")
else:
    print(f"Accuracy {accuracy} is below threshold (0.85). Model not registered.")
