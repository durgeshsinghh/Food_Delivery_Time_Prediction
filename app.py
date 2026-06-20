from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
from sklearn import set_config
import pandas as pd
import joblib
import uvicorn
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime

# Output as pandas
set_config(transform_output="pandas")


# --------------------- Input Schema ---------------------
class Data(BaseModel):
    Delivery_person_Age: float
    Delivery_person_Ratings: float
    Restaurant_latitude: float
    Restaurant_longitude: float
    Delivery_location_latitude: float
    Delivery_location_longitude: float
    Order_Date: str
    Time_Orderd: str
    Time_Order_picked: str
    Weatherconditions: str
    Road_traffic_density: str
    Vehicle_condition: int
    Type_of_order: str
    Type_of_vehicle: str
    multiple_deliveries: float
    Festival: str
    City: str


# --------------------- Load Model ---------------------
model = joblib.load("models/model.joblib")
preprocessor = joblib.load("models/preprocessor.joblib")

model_pipe = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("regressor", model)
    ]
)


# --------------------- Feature Engineering ---------------------
def create_features(data):

    # Distance calculation (Haversine formula)
    R = 6371

    lat1 = radians(data["Restaurant_latitude"])
    lon1 = radians(data["Restaurant_longitude"])

    lat2 = radians(data["Delivery_location_latitude"])
    lon2 = radians(data["Delivery_location_longitude"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2

    distance = 2 * R * atan2(sqrt(a), sqrt(1 - a))

    # Distance Type
    if distance < 5:
        distance_type = "Short"
    elif distance < 15:
        distance_type = "Medium"
    else:
        distance_type = "Long"

    # Pickup Time in Minutes
    fmt = "%H:%M:%S"

    t1 = datetime.strptime(data["Time_Orderd"], fmt)
    t2 = datetime.strptime(data["Time_Order_picked"], fmt)

    pickup_time_minutes = int((t2 - t1).seconds / 60)

    # Weekend
    order_date = pd.to_datetime(data["Order_Date"], dayfirst=True)
    is_weekend = int(order_date.weekday() >= 5)

    # Order Time of Day
    hour = t1.hour

    if hour < 12:
        order_time_of_day = "Morning"
    elif hour < 17:
        order_time_of_day = "Afternoon"
    elif hour < 21:
        order_time_of_day = "Evening"
    else:
        order_time_of_day = "Night"

    # Final feature dictionary
    features = {
        "age": data["Delivery_person_Age"],
        "ratings": data["Delivery_person_Ratings"],
        "weather": data["Weatherconditions"].replace("conditions ", "").strip(),
        "traffic": data["Road_traffic_density"].strip(),
        "vehicle_condition": int(data["Vehicle_condition"]),
        "type_of_order": data["Type_of_order"].strip(),
        "type_of_vehicle": data["Type_of_vehicle"].strip(),
        "multiple_deliveries": int(data["multiple_deliveries"]),
        "festival": data["Festival"].strip(),
        "city_type": data["City"].strip(),
        "is_weekend": is_weekend,
        "pickup_time_minutes": pickup_time_minutes,
        "order_time_of_day": order_time_of_day,
        "distance": round(distance, 2),
        "distance_type": distance_type
    }

    return features


# --------------------- FastAPI ---------------------
app = FastAPI()


@app.get("/")
def home():
    return {
        "message": "Welcome to Food Delivery Time Prediction API"
    }


@app.post("/predict")
def do_predictions(data: Data):
    try:

        raw_data = data.dict()

        features = create_features(raw_data)

        pred_df = pd.DataFrame([features])

        prediction = model_pipe.predict(pred_df)[0]

        return {
            "predicted_delivery_time": round(float(prediction), 2)
        }

    except Exception as e:
        return {
            "error": str(e)
        }


# --------------------- Run Server ---------------------
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )