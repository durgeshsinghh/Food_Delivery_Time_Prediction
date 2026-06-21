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
import subprocess

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
        # The model (~295MB serialized) is too large to reliably upload
        # as an MLflow artifact on DagsHub's hosted server -- uploads of
        # this size were timing out / returning 500 errors during testing.
        #
        # Instead of mlflow.sklearn.log_model() (which would re-serialize
        # and upload the full model), we keep the model versioned by DVC
        # (models/model.joblib is already a DVC-tracked output of an
        # earlier pipeline stage) and only record a lightweight pointer
        # to it in MLflow + run_information.json. register_model.py uses
        # this pointer with the model registry's create_model_version()
        # API directly, rather than mlflow.register_model()/runs:/ URIs,
        # since there's no "logged model" artifact for it to resolve.
        # --------------------------------------------------------------
        artifact_path = "delivery_time_pred_model"

        # Canonical registered model name -- must match the name used in
        # scripts/promote_model.py and scripts/register_model.py, otherwise
        # promote_model.py looks up a different (non-existent) registered
        # model and finds zero versions in Staging.
        model_name = "Food_Delivery_Time_Predictor"

        # record where DVC is tracking the actual model binary, plus the
        # git commit so the exact model version is reproducible/traceable
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root_path
            ).decode().strip()
        except Exception:
            git_commit = "unknown"

        dvc_model_path = str((Path("models") / "model.joblib").as_posix())

        mlflow.set_tag("dvc_model_path", dvc_model_path)
        mlflow.set_tag("git_commit", git_commit)
        mlflow.log_param("model_storage", "dvc")

        logger.info(f"Model is DVC-tracked at '{dvc_model_path}' (commit {git_commit}); "
                    f"skipping MLflow artifact upload due to size (~295MB)")

        run_id = run.info.run_id

        # save run info -- includes DVC pointer so register_model.py can
        # build a model_version source without needing an MLflow artifact
        run_info_full = {
            "run_id": run_id,
            "artifact_path": artifact_path,
            "model_name": model_name,
            "dvc_model_path": dvc_model_path,
            "git_commit": git_commit
        }
        with open(root_path / "run_information.json", "w") as f:
            json.dump(run_info_full, f, indent=4)

        logger.info("Model information saved")

        logger.info("MLflow logging complete")