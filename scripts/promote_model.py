import os
import dagshub
import mlflow
from mlflow import MlflowClient

# Authenticate using DAGSHUB_TOKEN (same pattern as evaluation.py /
# register_model.py) instead of letting dagshub.init() fall back to an
# interactive OAuth browser flow, which fails in CI (no browser, no human
# to click the authorization link) and crashes with a JSONDecodeError
# when the server returns a non-JSON error response.
username = "durgeshsinghh"
token = os.environ["DAGSHUB_TOKEN"]

os.environ["MLFLOW_TRACKING_USERNAME"] = username
os.environ["MLFLOW_TRACKING_PASSWORD"] = token

dagshub.auth.add_app_token(token)

dagshub.init(
    repo_owner="durgeshsinghh",
    repo_name="food_delivery_time_prediction",
    mlflow=True
)

# Set tracking URI
mlflow.set_tracking_uri(
    f"https://{username}:{token}@dagshub.com/durgeshsinghh/food_delivery_time_prediction.mlflow"
)

# Create client
client = MlflowClient()

model_name = "Food_Delivery_Time_Predictor"

# Get latest model in Staging.
# get_latest_versions() (stage-based) is deprecated in newer MLflow
# versions in favor of search_model_versions() with alias-based lookup.
# Try the legacy call first, fall back if unavailable.
try:
    latest_versions = client.get_latest_versions(
        name=model_name,
        stages=["Staging"]
    )
    latest_version = latest_versions[0].version
except AttributeError:
    # Newer MLflow versions removed get_latest_versions/stages
    versions = client.search_model_versions(f"name='{model_name}'")
    staging_versions = [
        v for v in versions
        if "staging" in getattr(v, "aliases", [])
    ]
    if not staging_versions:
        raise RuntimeError(
            f"No model version found with alias 'staging' for '{model_name}'"
        )
    latest_version = max(staging_versions, key=lambda v: int(v.version)).version


try:
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage="Production",
        archive_existing_versions=True
    )
except AttributeError:
    client.set_registered_model_alias(
        name=model_name,
        alias="production",
        version=latest_version
    )

print(
    f"Successfully promoted {model_name} Version {latest_version} to Production"
)
