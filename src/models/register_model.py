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
    dvc_model_path = run_info.get("dvc_model_path", "models/model.joblib")
    git_commit = run_info.get("git_commit", "unknown")

    # ------------------------------------------------------------------
    # The model is NOT logged as an MLflow artifact (it's ~295MB, which
    # was too large to reliably upload to DagsHub's MLflow artifact
    # store -- uploads were timing out / returning 500 errors). Instead
    # it is versioned by DVC. We register it in the MLflow Model
    # Registry using create_model_version() with a `source` that points
    # at the DVC-tracked path rather than a runs:/ URI, since there is
    # no logged-model artifact for mlflow.register_model() to resolve.
    #
    # `source` here records WHERE the model lives (DVC path), and
    # `run_id` links the registry entry back to the metrics/params for
    # this run for lineage purposes.
    # ------------------------------------------------------------------
    client = MlflowClient()

    # ensure the registered model exists (create_model_version requires this)
    try:
        client.create_registered_model(model_name)
        logger.info(f"Created new registered model '{model_name}'")
    except mlflow.exceptions.MlflowException:
        # already exists -- fine
        logger.info(f"Registered model '{model_name}' already exists")

    source = f"dvc://{dvc_model_path}?commit={git_commit}"
    logger.info(f"Registering model version with source: {source}")

    model_version = client.create_model_version(
        name=model_name,
        source=source,
        run_id=run_id,
        description=f"DVC-tracked model at commit {git_commit}"
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
