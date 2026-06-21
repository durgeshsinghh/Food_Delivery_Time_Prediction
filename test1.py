
"""
Run locally to verify create_model_version() accepts a non-runs:/ source
on DagsHub's registry before relying on it in CI.

Usage:
    set DAGSHUB_TOKEN=your_rotated_token
    python test_register.py
"""
import os
import mlflow
from mlflow import MlflowClient

username = "durgeshsinghh"
token = os.environ["DAGSHUB_TOKEN"]

mlflow.set_tracking_uri(
    f"https://{username}:{token}@dagshub.com/{username}/Food_Delivery_Time_Prediction.mlflow"
)
mlflow.set_experiment("DVC Pipeline")

client = MlflowClient()

TEST_MODEL_NAME = "diagnostic_dvc_register_test"
TEST_SOURCE = "dvc://models/model.joblib?commit=test123"

# need a run_id to link to -- create a throwaway one
with mlflow.start_run(run_name="register-test") as run:
    run_id = run.info.run_id
    mlflow.log_param("test", "register_diagnostic")

print("Run ID:", run_id)

try:
    client.create_registered_model(TEST_MODEL_NAME)
    print(f"Created registered model '{TEST_MODEL_NAME}'")
except mlflow.exceptions.MlflowException as e:
    print(f"Registered model already exists or error: {e}")

print(f"\nAttempting create_model_version with source='{TEST_SOURCE}'...")
try:
    mv = client.create_model_version(
        name=TEST_MODEL_NAME,
        source=TEST_SOURCE,
        run_id=run_id,
        description="diagnostic test"
    )
    print("SUCCESS. Version:", mv.version)
    print("Source stored as:", mv.source)
except Exception as e:
    print("FAILED:", e)
