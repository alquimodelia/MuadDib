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
