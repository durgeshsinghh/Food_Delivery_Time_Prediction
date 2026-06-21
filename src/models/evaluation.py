import pandas as pd
import joblib
import logging
import mlflow
import dagshub
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import numpy as np
import os

import mlflow.sklearn

# ---------------------------------------------------------------------------
# DagsHub / MLflow tracking setup
# ---------------------------------------------------------------------------
username = "durgeshsinghh"
token = os.environ["DAGSHUB_TOKEN"]

mlflow.set_tracking_uri(
    f"https://{username}:{token}@dagshub.com/{username}/Food_Delivery_Time_Prediction.mlflow"
)

mlflow.set_experiment("DVC Pipeline")

TARGET = "time_taken"

# create logger
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.INFO)

# console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

# formatter
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)


def load_data(data_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        logger.error("The file to load does not exist")
        raise
    return df


def make_X_and_y(data: pd.DataFrame, target_column: str):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y


def load_model(model_path: Path):
    model = joblib.load(model_path)
    return model


def save_model_info(save_json_path, run_id, artifact_path, model_name):
    info_dict = {
        "run_id": run_id,
        "artifact_path": artifact_path,
        "model_name": model_name
    }
    with open(save_json_path, "w") as f:
        json.dump(info_dict, f, indent=4)


if __name__ == "__main__":
    # root path
    root_path = Path(__file__).parent.parent.parent

    # data paths
    train_data_path = root_path / "data" / "processed" / "train_trans.csv"
    test_data_path = root_path / "data" / "processed" / "test_trans.csv"

    # model path
    model_path = root_path / "models" / "model.joblib"

    # load the training data
    train_data = load_data(train_data_path)
    logger.info("Train data loaded successfully")

    # load the test data
    test_data = load_data(test_data_path)
    logger.info("Test data loaded successfully")

    # split the train and test data
    X_train, y_train = make_X_and_y(train_data, TARGET)
    X_test, y_test = make_X_and_y(test_data, TARGET)
    logger.info("Data split completed")

    # load the model
    model = load_model(model_path)
    logger.info("Model loaded successfully")

    # get the predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    logger.info("Prediction on data complete")

    # calculate the train and test mae
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    logger.info("Error calculated")

    # calculate the r2 scores
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    logger.info("R2 score calculated")

    # calculate cross val scores
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )
    logger.info("Cross validation complete")

    # mean cross val score
    mean_cv_score = -(cv_scores.mean())

    # ------------------------------------------------------------------
    # MLflow logging
    # ------------------------------------------------------------------
    with mlflow.start_run() as run:
        # set tags
        mlflow.set_tag("model", "Food Delivery Time Regressor")

        # log parameters
        mlflow.log_params(model.get_params())

        # log metrics
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("mean_cv_score", mean_cv_score)

        # log individual cv scores
        mlflow.log_metrics(
            {f"CV_{num}": score for num, score in enumerate(-cv_scores)}
        )

        # log input datasets
        train_data_input = mlflow.data.from_pandas(train_data, targets=TARGET)
        test_data_input = mlflow.data.from_pandas(test_data, targets=TARGET)

        mlflow.log_input(dataset=train_data_input, context="training")
        mlflow.log_input(dataset=test_data_input, context="validation")

        # final evaluation metrics on test set
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        run_info = {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2_score": float(r2)
        }

        # save metrics json (temporary, will be overwritten below with run info)
        with open(root_path / "run_information.json", "w") as f:
            json.dump(run_info, f, indent=4)
        logger.info("run_information.json created successfully")

        # --------------------------------------------------------------
        # Log the model properly so it can be registered later.
        # mlflow.log_artifact() only copies a raw file and does NOT
        # create a registrable "logged model" entry. mlflow.sklearn.log_model()
        # (or the appropriate flavor) is required for mlflow.register_model()
        # to find it via runs:/<run_id>/<artifact_path>.
        # --------------------------------------------------------------
        artifact_path = "delivery_time_pred_model"
        logger.info("Logging model to MLflow")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path
        )
        logger.info("Model logged successfully")

        run_id = run.info.run_id
        model_name = artifact_path

        # save run info (single, consistent write)
        save_model_info(
            save_json_path=root_path / "run_information.json",
            run_id=run_id,
            artifact_path=artifact_path,
            model_name=model_name
        )
        logger.info("Model information saved")

        logger.info("MLflow logging complete")