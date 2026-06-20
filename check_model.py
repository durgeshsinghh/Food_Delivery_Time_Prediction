
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

models = client.search_registered_models()

for model in models:
    print(model.name)