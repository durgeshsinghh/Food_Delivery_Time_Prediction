import dagshub
import mlflow
from mlflow import MlflowClient

dagshub.init(
    repo_owner="durgeshsinghh",
    repo_name="food_delivery_time_prediction",
    mlflow=True
)

mlflow.set_tracking_uri(
    "https://dagshub.com/durgeshsinghh/food_delivery_time_prediction.mlflow"
)

client = MlflowClient()

client.transition_model_version_stage(
    name="Food_Delivery_Time_Predictor",
    version=1,
    stage="Staging"
)

print("Moved Version 1 to Staging")