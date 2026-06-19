import mlflow
import mlflow.sklearn
import joblib
import dagshub

dagshub.init(
    repo_owner="durgeshsinghh",
    repo_name="Food_Delivery_Time_Prediction",
    mlflow=True
)

mlflow.set_tracking_uri(
    "https://dagshub.com/durgeshsinghh/Food_Delivery_Time_Prediction.mlflow"
)

model = joblib.load("models/model.joblib")

print(type(model))

mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="delivery_time_pred_model",
    serialization_format="pickle"
)