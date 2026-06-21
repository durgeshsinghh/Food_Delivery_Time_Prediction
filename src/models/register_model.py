import mlflow
import dagshub
import json
from pathlib import Path
from mlflow import MlflowClient
import logging
import os

# create logger
logger = logging.getLogger("register_model")
logger.setLevel(logging.INFO)

# console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

# formatter
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# ---------------------------------------------------------------------------
# DagsHub / MLflow tracking setup
# ---------------------------------------------------------------------------
username = "durgeshsinghh"
token = os.environ["DAGSHUB_TOKEN"]

mlflow.set_tracking_uri(
    f"https://{username}:{token}@dagshub.com/{username}/Food_Delivery_Time_Prediction.mlflow"
)

mlflow.set_experiment("DVC Pipeline")


def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
    return run_info


if __name__ == "__main__":
    # root path
    root_path = Path(__file__).parent.parent.parent

    # run information file path
    run_info_path = root_path / "run_information.json"

    # load run info written by evaluation.py
    run_info = load_model_information(run_info_path)

    run_id = run_info["run_id"]
    model_name = run_info["model_name"]
    artifact_path = run_info["artifact_path"]  # e.g. "delivery_time_pred_model"

    # model URI must point at the artifact_path used in mlflow.sklearn.log_model(),
    # NOT at a raw filename like model.joblib
    model_registry_path = f"runs:/{run_id}/{artifact_path}"

    logger.info(f"Registering model from {model_registry_path}")

    # register the model
    model_version = mlflow.register_model(
        model_uri=model_registry_path,
        name=model_name
    )

    registered_model_version = model_version.version
    registered_model_name = model_version.name
    logger.info(f"The latest model version in model registry is {registered_model_version}")

    # ------------------------------------------------------------------
    # Move model to "Staging".
    # client.transition_model_version_stage() is deprecated in newer
    # MLflow versions. set_registered_model_alias() is the supported
    # replacement, but we try the legacy call first for backward
    # compatibility, then fall back to aliases if unavailable.
    # ------------------------------------------------------------------
    client = MlflowClient()

    try:
        client.transition_model_version_stage(
            name=registered_model_name,
            version=registered_model_version,
            stage="Staging"
        )
        logger.info("Model pushed to Staging stage")
    except AttributeError:
        # Newer MLflow versions removed transition_model_version_stage
        client.set_registered_model_alias(
            name=registered_model_name,
            alias="staging",
            version=registered_model_version
        )
        logger.info("Model tagged with alias 'staging' (stages API unavailable)")
