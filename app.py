from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
import uvicorn
import pandas as pd
import joblib
from sklearn import set_config

# Output as pandas
set_config(transform_output='pandas')


class Data(BaseModel):
    age: float
    ratings: float
    weather: str
    traffic: str
    vehicle_condition: int
    type_of_order: str
    type_of_vehicle: str
    multiple_deliveries: int
    festival: str
    city_type: str
    is_weekend: int
    pickup_time_minutes: int
    order_time_of_day: str
    distance: float
    distance_type: str


# Load model directly from local file
model = joblib.load("models/model.joblib")

# Load preprocessor
preprocessor = joblib.load("models/preprocessor.joblib")

# Build pipeline
model_pipe = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("regressor", model)
    ]
)

# Create FastAPI app
app = FastAPI()


@app.get("/")
def home():
    return {
        "message": "Welcome to Food Delivery Time Prediction API"
    }


@app.post("/predict")
def do_predictions(data: Data):

    pred_data = pd.DataFrame([data.dict()])

    prediction = model_pipe.predict(pred_data)[0]

    return {
        "predicted_delivery_time": round(float(prediction), 2)
    }


    # Prediction
    prediction = model_pipe.predict(pred_data)[0]

    return {
        "predicted_delivery_time": round(float(prediction), 2)
    }


if __name__ == "__main__":
    uvicorn.run(
        app="app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )