import os
import pathlib
import sys

import dotenv


def startup(env_file=".env"):
    dotenv.load_dotenv(env_file)
    PROJECT_FOLDER = os.getenv("PROJECT_FOLDER", None)
    if PROJECT_FOLDER is None:
        PROJECT_FOLDER = "."
    PROJECT_FOLDER = pathlib.Path(PROJECT_FOLDER).resolve()

    PROJECT_NAME = os.getenv("PROJECT_NAME", "muaddib_project")
    SCRIPTS_FOLDER = os.getenv("SCRIPTS_FOLDER", None)
    if SCRIPTS_FOLDER is None:
        SCRIPTS_FOLDER = os.path.join(PROJECT_FOLDER, PROJECT_NAME)
    os.environ["SCRIPTS_FOLDER"] = SCRIPTS_FOLDER

    MODELS_FOLDER = os.getenv("MODELS_FOLDER", None)
    if MODELS_FOLDER is None:
        MODELS_FOLDER = os.path.join(PROJECT_FOLDER, "models")
    os.environ["MODELS_FOLDER"] = MODELS_FOLDER

    NOTEBOOKS_FOLDER = os.getenv("NOTEBOOKS_FOLDER", None)
    if NOTEBOOKS_FOLDER is None:
        NOTEBOOKS_FOLDER = os.path.join(PROJECT_FOLDER, "notebooks")
    os.environ["NOTEBOOKS_FOLDER"] = NOTEBOOKS_FOLDER

    REFERENCES_FOLDER = os.getenv("REFERENCES_FOLDER", None)
    if REFERENCES_FOLDER is None:
        REFERENCES_FOLDER = os.path.join(PROJECT_FOLDER, "references")
    os.environ["REFERENCES_FOLDER"] = REFERENCES_FOLDER

    REPORTS_FOLDER = os.getenv("REPORTS_FOLDER", None)
    if REPORTS_FOLDER is None:
        REPORTS_FOLDER = os.path.join(PROJECT_FOLDER, "reports")
    os.environ["REPORTS_FOLDER"] = REPORTS_FOLDER

    EXPERIMENT_FOLDER = os.getenv("EXPERIMENT_FOLDER", None)
    if EXPERIMENT_FOLDER is None:
        EXPERIMENT_FOLDER = os.path.join(PROJECT_FOLDER, "experiments")
    os.environ["EXPERIMENT_FOLDER"] = EXPERIMENT_FOLDER

    DATA_FOLDER = os.getenv("DATA_FOLDER", None)
    if DATA_FOLDER is None:
        DATA_FOLDER = os.path.join(PROJECT_FOLDER, "data")
    os.environ["DATA_FOLDER"] = DATA_FOLDER

    sys.path.append(SCRIPTS_FOLDER)

    KERAS_BACKEND = os.getenv("KERAS_BACKEND", "torch")
    os.environ["KERAS_BACKEND"] = KERAS_BACKEND

    RAW_FILE_NAME = os.getenv("RAW_FILE_NAME")
    RAW_FILE_PATH = os.path.join(DATA_FOLDER, "raw", RAW_FILE_NAME)
    os.environ["RAW_FILE_PATH"] = RAW_FILE_PATH

    PROCESSED_FILE_NAME = os.getenv("PROCESSED_FILE_NAME")
    PROCESSED_FILE_PATH = os.path.join(
        DATA_FOLDER, "processed", PROCESSED_FILE_NAME
    )
    os.environ["PROCESSED_FILE_PATH"] = PROCESSED_FILE_PATH

    VALIDATION_TARGET = os.getenv("VALIDATION_TARGET", "EPEA")
    os.environ["VALIDATION_TARGET"] = VALIDATION_TARGET

    REDO_VALIDATION = os.getenv("REDO_VALIDATION", False)
    os.environ["REDO_VALIDATION"] = REDO_VALIDATION

    # One target var - Upward
    # List of target variables, multiples variable - Upward;Downward
    # List of experiment with diferent targest - Upward|Downward
    # Mulitple choise, multiple experiment - Upward;Downward|Tender
    #   one set of experiments with targets: Upward;Downward
    #   another set of experiment with target: Tender
    TARGET_VARIABLE = os.getenv("TARGET_VARIABLE")

    MLFLOW_ADRESS = os.getenv("MLFLOW_ADRESS", None)
    MLFLOW_PORT = os.getenv("MLFLOW_PORT", None)

    MLFLOW_STATE = os.getenv("MLFLOW_STATE", "off")
    os.environ["MLFLOW_STATE"] = MLFLOW_STATE
    if MLFLOW_STATE == "on":
        if MLFLOW_ADRESS is not None:
            if MLFLOW_PORT is not None:
                mlflow_startup(host=MLFLOW_ADRESS, port=MLFLOW_PORT)
                os.environ["MLFLOW_STATE"] = "on"
        # Set the MLFLOW_TRACKING_URI environment variable
        os.environ["MLFLOW_TRACKING_URI"] = str(
            PROJECT_FOLDER.joinpath("mlruns").as_uri()
        )


def mlflow_startup(host="127.0.0.1", port="8080"):
    # import mlflow
    # uri = "http://{host}:{port}"
    # mlflow.set_tracking_uri(uri)
    # try:
    #     experiments = mlflow.list_experiments()
    #     print("MLflow server is running.")
    # except Exception as e:
    #     print("MLflow server is not running.")
    #     command = f"mlflow server --host {host} --port {port}"
    #     subprocess.Popen(command, shell=True)
    pass
