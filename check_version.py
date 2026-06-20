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

print("Registered Models:")
for model in client.search_registered_models():
    print("\nModel:", model.name)

    versions = client.search_model_versions(
        f"name='{model.name}'"
    )

    if len(versions) == 0:
        print("No versions found.")
    else:
        for v in versions:
            print(
                f"Version={v.version}, Stage={v.current_stage}, RunID={v.run_id}"
            )