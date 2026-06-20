import dagshub
import mlflow
from mlflow import MlflowClient

# Initialize DagsHub
dagshub.init(
    repo_owner="durgeshsinghh",
    repo_name="food_delivery_time_prediction",
    mlflow=True
)

# Set tracking URI
mlflow.set_tracking_uri(
    "https://dagshub.com/durgeshsinghh/food_delivery_time_prediction.mlflow"
)

# Create client
client = MlflowClient()

model_name = "Food_Delivery_Time_Predictor"

# Get latest model in Staging
latest_versions = client.get_latest_versions(
    name=model_name,
    stages=["Staging"]
)

latest_version = latest_versions[0].version

# Promote to Production
client.transition_model_version_stage(
    name=model_name,
    version=latest_version,
    stage="Production",
    archive_existing_versions=True
)

print(
    f"Successfully promoted {model_name} Version {latest_version} to Production"
)