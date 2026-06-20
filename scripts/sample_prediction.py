import pandas as pd
import requests
from pathlib import Path

# path
root_path = Path(__file__).parent.parent
data_path = root_path / "data" / "raw" / "India-Food-Delivery-Time-Prediction.txt"

# endpoint
predict_url = "http://127.0.0.1:8000/predict"

sample_row = pd.read_json(data_path).dropna().sample(1)

actual_value = sample_row.iloc[:, -1].item()
print("Actual value:", actual_value)

# remove target column only
data = sample_row.drop(columns=[sample_row.columns[-1]]).iloc[0].to_dict()

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json=data
)

print(response.json())