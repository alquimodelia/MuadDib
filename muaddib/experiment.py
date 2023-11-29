# ShaiHulud
import glob
import importlib
import inspect
import itertools
import json
import os

import alquitable
import numpy as np
from keras_core.losses import MeanSquaredError

from muaddib.shaihulud_utils import (
    get_target_dict,
    load_json_dict,
    open_model,
    read_model_conf,
    write_dict_to_file,
)

MLFLOW_STATE = os.getenv("MLFLOW_STATE", "off")
VALIDATION_TARGET = os.getenv("VALIDATION_TARGET", "bscore")

if MLFLOW_STATE == "on":
    import mlflow
    from mlflow import MlflowClient

    adress = "127.0.0.1"
    port = "8080"
    MLFLOW_ADRESS = os.getenv("MLFLOW_ADRESS", None)
    MLFLOW_PORT = os.getenv("MLFLOW_PORT", None)
    adress = adress or MLFLOW_ADRESS
    port = port or MLFLOW_PORT

    tracking_uri = f"http://{adress}:{port}"
    client = MlflowClient(tracking_uri=tracking_uri)


# TODO: make some lightweight version of the object, like just the case.__dict__ to get str values out of it
def is_jsonable(x):
    if isinstance(x, list):
        return all(is_jsonable(item) for item in x)
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def get_mirror_weight_loss(loss_name):
    loss_used = loss_name.replace("mirror_weights_", "")
    from alquitable.advanced_losses import MirrorWeights
    from alquitable.losses import ALL_LOSSES_DICT

    if "mirror_weights" in loss_name:
        weight_on_surplus = True
        if "reversed" in loss_name:
            weight_on_surplus = False
        words = loss_used.split("_")
        words = [w.title() for w in words]
        loss_used = "".join(words)
    if len(loss_used.split("_")) > 1:
        words = loss_used.split("_")
        words = [f.title() for f in words]
        loss_used = "".join(words)
    loss_used_fn = ALL_LOSSES_DICT.get(loss_used, None)
    if loss_used_fn is None:
        from keras_core.src.losses import ALL_OBJECTS_DICT

        loss_used_fn = ALL_OBJECTS_DICT.get(loss_used, None)
    if loss_used_fn is None:
        print("loss not found")
        print(loss_name)
        print(loss_used)
        print("------------")
        return
    if "mirror_weights" in loss_name:
        loss_used_fn = MirrorWeights(
            loss_to_use=loss_used_fn(), weight_on_surplus=weight_on_surplus
        )
    else:
        loss_used_fn = loss_used_fn()

    return loss_used_fn


class SpiceEyes:
    def __init__(
        self,
        target_name=None,
        target_variables=None,
        work_folder=None,
        name=None,
        epochs=200,
        optimizer="adam",
        batch_size=252,
        loss=None,
        callbacks=None,
        metrics=None,
        train_fn=None,
        validation_fn=None,
        visualize_report_fn=None,
        keras_backend="torch",
        benchmark_score_file=None,
        conf_file=None,
        DataManager=None,
        get_models_on_experiment_fn=None,
        get_models_args=None,
    ):
        callbacks = callbacks or []
        metrics = metrics or ["root_mean_squared_error"]
        self.target_name = target_name
        self.target_variables = target_variables
        self.DataManager = DataManager

        self.name = name
        self.work_folder = work_folder
        self.epochs = epochs
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.callbacks = callbacks
        self.loss = loss or MeanSquaredError()
        self.metrics = metrics
        self.train_fn = train_fn
        self.validation_fn = validation_fn
        self.visualize_report_fn = visualize_report_fn
        self.keras_backend = keras_backend
        self.complete = False
        self.predict_score = {}
        self.benchmark_score_file = benchmark_score_file
        self.worthy_models = None
        self.conf_file = conf_file
        self.predict_score_path = None
        if self.conf_file:
            if os.path.exists(self.conf_file):
                self.__dict__ = self.load(self.conf_file)

    def set_benchmark_path(self):
        if self.benchmark_score_file is None:
            DATA_FOLDER = os.getenv("DATA_FOLDER", "data")
            benchmark_data_folder = os.path.join(
                DATA_FOLDER, "benchmark", self.DataManager.name
            )
            self.benchmark_score_file = os.path.join(
                benchmark_data_folder, "benchmark.json"
            )

    def set_predict_score(self, new_predict_score):
        for key in new_predict_score.keys():
            value = new_predict_score[key]
            if not isinstance(value, list):
                value = [value]
            if key not in self.predict_score:
                self.predict_score[key] = value
            else:
                self.predict_score[key] += value

    def save(self, path=None):
        dict_to_load = self.__dict__.copy()
        dict_to_save = {}
        for key, value in dict_to_load.items():
            dict_to_save[key] = value
            if value is None:
                continue

            if key == "predict_score":
                # Read from path on load (self.predict_score_path)
                dict_to_save[key] = self.predict_score_path

            if key == "conf":
                dict_to_save[key] = list(range(len(dict_to_load[key])))
                for i, k in enumerate(dict_to_load[key]):
                    dict_to_save[key][i] = k.conf_file
            if key == "study_cases":
                dict_to_save[key] = {}
                for k, v in dict_to_load[key].items():
                    dict_to_save[key][k] = v.conf_file
            if key == "models_dict":
                dict_to_save[key] = {}
                for k in dict_to_load[key].keys():
                    dict_to_save[key][k] = os.path.join(
                        os.getenv("MODELS_FOLDER"), f"{k}.json"
                    )
            if "_fn" in key:  # function
                # TODO: check if we can load then with this or if its the model that should be saved
                dict_to_save[key] = f"{value.__module__}:{value.__name__}"

            if key == "callbacks":
                value_str = [
                    str(f).split(".")[-1].replace("'>", "")
                    for f in self.callbacks
                ]
                dict_to_save[key] = value_str
            if key == "previous_cases":
                if dict_to_load[key]:
                    dict_to_save[key] = list(range(len(dict_to_load[key])))
                    for i, k in enumerate(dict_to_load[key]):
                        dict_to_save[key][i] = f"{k.experiment_name}:{k.name}"
            if key == "worthy_cases":
                dict_to_save[key] = [f.name for f in value]

            if key == "loss":
                if dict_to_load[key]:
                    value_str = self.loss
                    if not isinstance(value_str, list):
                        value_str = [value_str]
                    dict_to_save[key] = list(range(len(value_str)))
                    for i, k in enumerate(value_str):
                        dict_to_save[key][i] = getattr(k, "name", k)
            if not is_jsonable(dict_to_save[key]):
                dict_to_save[key] = (
                    dict_to_save[key].name
                    if hasattr(dict_to_save[key], "name")
                    else str(dict_to_save[key])
                )

        with open(path, "w") as f:
            json.dump(dict_to_save, f)

    @staticmethod
    def load(path=None):
        with open(path, "r") as f:
            dict_to_load = json.load(f)
        # Create a new dictionary to store the loaded data
        dict_to_restore = {}

        # Restore the original data structures and objects
        for key, value in dict_to_load.items():
            dict_to_restore[key] = value
            if value is None:
                continue
            if key == "conf":
                # Load a Case
                dict_to_restore[key] = [
                    Case(conf_file=conf_file) for conf_file in value
                ]
            elif key == "study_cases":
                # Load a Case
                # TODO: there are problems when loading a case that was deleted, it does not do it again.
                dict_to_restore[key] = {
                    k: Case(conf_file=v) for k, v in value.items()
                }
            elif key == "models_dict":
                dict_to_restore[key] = {
                    k: read_model_conf(v) for k, v in value.items()
                }
            elif "_fn" in key:
                module_name, function_name = value.split(":")
                module = importlib.import_module(module_name)
                dict_to_restore[key] = getattr(module, function_name)
            elif key == "callbacks":
                dict_to_restore[key] = [
                    getattr(alquitable.callbacks, callback_name)
                    for callback_name in value
                ]
            elif key == "previous_cases":
                previous_cases = []
                experiments_in_list = []
                for f in value:
                    target_name, exp_name, case_name = f.split(":")
                    experiments_in_list.append(f"{target_name}:{exp_name}")

                experiments_in_list = np.unique(experiments_in_list)
                cases_per_experiment = {}
                for exp in experiments_in_list:
                    cases_per_experiment[exp] = []
                    for f in value:
                        target_name, exp_name, case_name = f.split(":")
                        if f"{target_name}:{exp_name}" == exp:
                            cases_per_experiment[exp].append(case_name)
                    folder_target, folder_exp = exp.split(":")

                    path_experiment = os.path.join(
                        os.getenv("EXPERIMENT_FOLDER"),
                        folder_target,
                        folder_exp,
                        "exp_conf.json",
                    )
                    exp_obj = Experiment(conf_file=path_experiment)
                    previous_cases += [
                        case_obj
                        for case_name, case_obj in exp_obj.study_cases.items()
                        if case_name in cases_per_experiment[exp]
                    ]
                dict_to_restore[key] = previous_cases
            elif key == "loss":
                dict_to_restore[key] = [
                    get_mirror_weight_loss(loss_name) for loss_name in value
                ]
            elif key == "predict_score":
                if os.path.exists(str(value)):
                    dict_to_restore[key] = load_json_dict(value)
                else:
                    dict_to_restore[key] = {}
            elif key == "previous_experiment":
                folder_target, folder_exp = value.split(":")
                path_experiment = os.path.join(
                    os.getenv("EXPERIMENT_FOLDER"),
                    folder_target,
                    folder_exp,
                    "exp_conf.json",
                )
                exp_obj = Experiment(conf_file=path_experiment)
                dict_to_restore[key] = exp_obj
            elif key == "DataManager":
                from data.definitions import ALL_DATA_MANAGERS

                dict_to_restore[key] = ALL_DATA_MANAGERS[value]

            # elif key == "model":
            #     pass

        if "worthy_cases" in dict_to_restore:
            if len(dict_to_restore["worthy_cases"]) > 0:
                dict_to_restore["worthy_cases"] = [
                    case_obj
                    for case_name, case_obj in dict_to_restore[
                        "study_cases"
                    ].items()
                    if case_name in dict_to_restore["worthy_cases"]
                ]

        return dict_to_restore
        # Update the object's dictionary with the loaded data
        # self.__dict__.update(dict_to_restore)


class Case(SpiceEyes):
    def __init__(
        self,
        work_folder=None,
        case_name="",  # Case specific
        model_name="",  # Model Name
        model=None,
        freq_saves="freq_saves",
        model_types=".keras",
        model_conf_name="model_conf.json",
        experiment_name="",
        previous_benchmark=None,
        conf_file=None,
        **kwargs,
    ):
        self.case_name = case_name
        self.model_name = model_name

        self.model = model
        self.freq_saves = freq_saves
        self.model_types = model_types
        self.model_conf_name = model_conf_name
        self.experiment_name = experiment_name
        self.some_training = False
        self.best_result = None
        self.previous_benchmark = previous_benchmark
        self.worthy = False
        super().__init__(work_folder=work_folder, **kwargs)
        if conf_file is not None:
            if os.path.exists(conf_file):
                self.__dict__.update(self.load(conf_file))

        self.setup()

    def check_trained_epochs(self):
        # Checks how many epochs were trained
        list_query = f"{self.case_work_frequency_path}/**{self.model_types}"
        list_freq_saves = glob.glob(list_query)
        self.list_freq_saves = list_freq_saves
        last_epoch = 0
        last_epoch_path = None
        if len(list_freq_saves) > 0:
            epocs_done = [
                int(os.path.basename(f).replace(self.model_types, ""))
                for f in list_freq_saves
            ]
            last_epoch = max(epocs_done)
            last_epoch_path = f"{self.case_work_frequency_path}/{last_epoch}{self.model_types}"
        self.last_epoch_path = last_epoch_path
        self.last_epoch = last_epoch
        return

    def setup(
        self,
    ):
        if self.name is None:
            self.name = self.model_name

        if self.case_name:
            self.name = f"{self.name}_{self.case_name}"

        self.case_work_path = os.path.join(self.work_folder, self.name)
        self.model_keras_path = os.path.join(
            self.case_work_path, f"{self.model_name}.keras"
        )
        self.predict_score_path = os.path.join(
            self.case_work_path, "predict_score.json"
        )

        # Frequency saves
        self.case_work_frequency_path = os.path.join(
            self.case_work_path, self.freq_saves
        )
        self.conf_file = self.conf_file or os.path.join(
            self.case_work_path, "case_conf.json"
        )
        self.set_benchmark_path()
        if os.path.exists(self.conf_file):
            self.__dict__.update(self.load(self.conf_file))
        else:
            os.makedirs(self.case_work_path, exist_ok=True)
            os.makedirs(self.case_work_frequency_path, exist_ok=True)
            self.save(self.conf_file)
        self.set_fit_args()
        self.set_compile_args()
        self.load_model()

    def set_compile_args(self):
        compile_args = {
            "optimizer": self.optimizer,
            "loss": self.loss,
            "metrics": self.metrics,
        }

        self.compile_args = compile_args

    def set_fit_args(self):
        self.check_trained_epochs()
        epocs_to_train = self.epochs - self.last_epoch
        if epocs_to_train < self.epochs:
            self.some_training = True

        if epocs_to_train < 1:
            self.complete = True
            return

        callbacks = []
        if not isinstance(self.callbacks, list):
            self.callbacks = [self.callbacks]

        for callback in self.callbacks:
            qry = f"{self.case_work_path}/**.json"
            json_list = glob.glob(qry)
            if not isinstance(json_list, list):
                json_list = [json_list]
            # TODO: Wtf dude? is this suposed to be the model_name.json? maybe not needed at all
            json_list = [f for f in json_list if "_conf" not in f]
            json_list = [f for f in json_list if "score" not in f]

            if len(json_list) == 0:
                json_list = None
            else:
                json_list = json_list[0]

            callback_args = {}
            arg_names = inspect.getfullargspec(callback).args
            if "save_frequency" in arg_names:
                callback_args["save_frequency"] = 1
            if "start_epoch" in arg_names:
                callback_args["start_epoch"] = self.last_epoch
            if "model_keras_filename" in arg_names:
                frq_model_filename_sof = (
                    f"{self.case_work_frequency_path}" + "/{epoch}.keras"
                )
                callback_args["model_keras_filename"] = frq_model_filename_sof
            if "filepath" in arg_names:
                callback_args["filepath"] = self.model_keras_path
            if "model_log_filename" in arg_names:
                callback_args[
                    "model_log_filename"
                ] = json_list or self.model_keras_path.replace(
                    ".keras", ".json"
                )
            if "logs" in arg_names:
                model_history_filename = (
                    json_list
                    or self.model_keras_path.replace(".keras", ".json")
                )
                if os.path.exists(model_history_filename):
                    with open(model_history_filename) as f:
                        history = json.load(f)
                    callback_args["logs"] = history
            callbacks.append(callback(**callback_args))

        fit_args = {
            "epochs": epocs_to_train,
            "callbacks": callbacks,
            "batch_size": self.batch_size,
        }

        self.fit_args = fit_args

    def load_model(self):
        custom_objects = {"loss": self.loss}
        if self.last_epoch_path:
            self.model = open_model(
                self.last_epoch_path, custom_objects=custom_objects
            )
        elif isinstance(self.model, str):
            if os.path.exists(self.model):
                self.model = open_model(self.model)
            else:
                path_to_model_conf = os.path.join(
                    os.getenv("MODELS_FOLDER"), f"{self.model}.json"
                )
                self.model = open_model(path_to_model_conf)
        elif self.model is None:
            self.model = open_model(self.model_keras_path)

    def set_mlflow(self):
        if MLFLOW_STATE != "on":
            return
        # Sets the current active experiment to the "Apple_Models" experiment and
        # returns the Experiment metadata
        # experiment = mlflow.set_experiment(self.experiment_name)
        mlflow.set_experiment(self.experiment_name)

        # Define a run name for this iteration of training.
        # If this is not set, a unique name will be auto-generated for your run.

    def train_model(self):
        if self.complete:
            return
        # if os.getenv("MLFLOW_STATE")=="on":
        # if True:

        #     from mlflow import MlflowClient
        #     client = MlflowClient(tracking_uri="http://127.0.0.1:8080")

        # Start a new MLflow run
        if MLFLOW_STATE == "on":
            mlflow.set_tracking_uri("http://127.0.0.1:8080")

            self.set_mlflow()
            run = mlflow.start_run(run_name=self.name)
            mlflow_callback = mlflow.keras_core.MLflowCallback(run)
            if mlflow_callback not in self.fit_args["callbacks"]:
                self.fit_args["callbacks"].append(mlflow_callback)
        print("-----------------------------------------------------------")
        print(f"Training Model {self.name} from {self.experiment_name}")
        self.train_fn(
            self.model,
            self.DataManager,
            fit_args=self.fit_args,
            compile_args=self.compile_args,
            model_name=self.name,
        )
        if MLFLOW_STATE == "on":
            # End the MLflow run
            mlflow.end_run()

    def validate_model(self):
        if len(self.list_freq_saves) == 0:
            self.setup()

        # TODO: Get it also some env to do this if necessary
        REDO_VALIDATION = os.getenv("REDO_VALIDATION", False)
        do_validation = REDO_VALIDATION
        if not self.predict_score:
            do_validation = True

        if True:
            for freq_save in self.list_freq_saves:
                epoch = int(
                    os.path.basename(freq_save).replace(self.model_types, "")
                )

                prediction_path = freq_save.replace(
                    "freq_saves", "freq_predictions"
                ).replace(".keras", ".npz")

                score_path = freq_save.replace(
                    "freq_saves", "freq_predictions"
                ).replace(".keras", ".json")

                bool_prediction = os.path.exists(prediction_path)
                bool_score = os.path.exists(score_path)
                validation_done = bool_prediction & bool_score
                # TODO: change this to a way to do a new validation if the scores change?
                # A variable input args in the command
                # if true does the score agina  if false just does the fihures
                predict_score = None
                # TODO: set env varibale for this with command input
                if True:
                    predict_score = self.validation_fn(
                        datamanager=self.DataManager,
                        model_path=freq_save,
                        model_name=self.name,
                        epoch=epoch,
                    )
                else:
                    if bool_score:
                        predict_score = load_json_dict(score_path)
                if predict_score:
                    if self.case_name:
                        predict_score["case"] = self.case_name
                self.set_predict_score(predict_score)
            write_dict_to_file(self.predict_score, self.predict_score_path)
        if self.some_training:
            # TODO also only do it if its not done or a env variables
            self.write_report()
            # self.save(self.conf_file)
        if not self.complete:
            return

    def write_report(self):
        import pandas as pd

        # TODO: change this path thingys
        case_report_path = self.case_work_path.replace(
            "experiments", "reports"
        )
        os.makedirs(case_report_path, exist_ok=True)
        case_results = pd.DataFrame(self.predict_score)

        COLUMN_TO_SORT_BY = VALIDATION_TARGET
        ascending_to_sort = False
        # All results, ordered by epoch
        case_results.sort_values(by="epoch", inplace=True)
        self.best_result = max(case_results[COLUMN_TO_SORT_BY])
        if self.previous_benchmark:
            self.worthy = self.best_result >= self.previous_benchmark
        path_schema_csv = os.path.join(
            case_report_path, f"case_results_{VALIDATION_TARGET}.csv"
        )
        path_schema_tex = os.path.join(
            case_report_path, f"case_results_{VALIDATION_TARGET}.tex"
        )
        case_results.sort_values(
            by=COLUMN_TO_SORT_BY, ascending=ascending_to_sort, inplace=True
        )
        if len(case_results) > 10:
            case_results_sort = pd.concat(
                [case_results.head(), case_results.tail()]
            )
            case_results_sort.to_csv(path_schema_csv, index=False)
            case_results_sort.to_latex(
                path_schema_tex, escape=False, index=False, float_format="%.2f"
            )
            path_schema_csv = os.path.join(
                case_report_path,
                f"case_results_{VALIDATION_TARGET}_complete.csv",
            )
            path_schema_tex = os.path.join(
                case_report_path,
                f"case_results_{VALIDATION_TARGET}_complete.tex",
            )

        case_results.to_csv(path_schema_csv, index=False)
        case_results.to_latex(
            path_schema_tex, escape=False, index=False, float_format="%.2f"
        )


class Experiment(SpiceEyes):
    def __init__(
        self,
        name=None,
        work_folder=None,
        models_dict=None,
        description="",
        experiment_tags=None,
        previous_experiment=None,
        previous_cases=None,
        get_models_on_experiment_fn=None,
        get_models_args=None,
        conf_file=None,
        forward_epochs=None,
        **kwargs,
    ):
        self.models_dict = models_dict
        self.previous_experiment = previous_experiment
        self.previous_cases = previous_cases
        self.best_result = None
        self.get_models_on_experiment_fn = get_models_on_experiment_fn
        self.get_models_args = get_models_args or {}
        self.forward_epochs = forward_epochs
        super().__init__(name=name, work_folder=work_folder, **kwargs)
        self.description = description
        self.set_tags(experiment_tags)
        if conf_file is not None:
            if os.path.exists(conf_file):
                self.__dict__.update(self.load(conf_file))
        # self.set_mlflow_experiment()
        if not self.previous_experiment:
            self.setup()
        # self.experiment_configuration(models_dict)

    def set_tags(self, experiment_tags=None):
        experiment_tags = experiment_tags or {}
        default_tags = {
            "project_name": os.getenv("PROJECT_NAME", "muaddib_project"),
            "mlflow.note.content": self.description,
        }

        default_tags.update(experiment_tags)
        self.experiment_tags = default_tags

    def set_mlflow_experiment(self):
        if MLFLOW_STATE != "on":
            return

        mlflow_experiment = client.search_experiments(
            filter_string="name = '{0}'".format(self.name)
        )
        if len(mlflow_experiment) > 0:
            self.mlflow_experiment = mlflow_experiment[0]
        else:
            self.mlflow_experiment = client.create_experiment(
                name=self.name, tags=self.experiment_tags
            )

    def setup(
        self,
    ):
        folder_name = self.name.split(":")[-1]
        self.case_work_path = os.path.join(
            self.work_folder, self.DataManager.name, folder_name
        )
        self.predict_score_path = os.path.join(
            self.case_work_path, "predict_score.json"
        )
        os.makedirs(self.case_work_path, exist_ok=True)
        if self.previous_experiment:
            self.previous_cases = self.previous_experiment.worthy_cases

        self.set_mlflow_experiment()
        self.conf_file = os.path.join(self.case_work_path, "exp_conf.json")
        self.set_benchmark_path()
        # TODO: dont even know, but the loading needs to change
        if os.path.exists(self.conf_file):
            self.__dict__.update(self.load(self.conf_file))
        else:
            if self.get_models_on_experiment_fn is not None:
                self.models_dict = self.get_models_on_experiment_fn(
                    self.previous_experiment.worthy_models,
                    input_args=self.get_models_args,
                )
            self.experiment_configuration(self.models_dict)
            self.save(self.conf_file)

    def get_compile_args(self, optimizer, loss, metrics):
        compile_args = {
            "optimizer": optimizer,
            "loss": loss,
            "metrics": metrics,
        }

        self.compile_args = compile_args
        return compile_args

    def get_fit_args(self, epochs, callbacks, batch_size):
        fit_args = {
            "epochs": epochs,
            "callbacks": callbacks,
            "batch_size": batch_size,
        }

        self.fit_args = fit_args
        return fit_args

    def case_configuration(
        self,
        model_name,
        optimizer=None,
        loss=None,
        metrics=None,
        epochs=None,
        callbacks=None,
        batch_size=None,
        model=None,
        weight=None,
        previous_benchmark=None,
        name=None,
        previous_case=None,
    ):
        case_list = []
        metrics = metrics or self.metrics
        callbacks = callbacks or self.callbacks
        epochs = epochs or self.epochs

        # Define a dictionary with the variables and their default values
        variables = {
            "optimizer": optimizer or self.optimizer,
            "loss": loss or self.loss,
            "batch_size": batch_size or self.batch_size,
        }

        vars_lists = []
        for key, value in variables.items():
            # Check if the value is a list
            if isinstance(value, list):
                # This give me a list of which keys are cases
                vars_lists.append(key)
            else:
                variables[key] = [value]

        # Extract the lists from the dictionary
        lists = list(variables.values())

        # Use itertools.product to get all combinations
        for combination in itertools.product(*lists):
            case_name = ""
            # Zip the keys with the combination
            args = {
                key: value for key, value in zip(variables.keys(), combination)
            }
            # get the case name
            for key, value in args.items():
                if key in vars_lists:
                    if isinstance(value, int):
                        key_name = value
                    elif isinstance(value, str):
                        key_name = value
                    else:
                        key_name = value.name
                        # Just the 1st letter of each word
                        key_name = "".join([f[0] for f in key_name.split("_")])

                    case_name += f"{key_name}_"
            if case_name.endswith("_"):
                case_name = case_name[:-1]
            case_obj = Case(
                **args,
                metrics=metrics,
                callbacks=callbacks,
                model_name=model_name,
                case_name=case_name,
                epochs=epochs,
                model=model,
                work_folder=self.case_work_path,
                train_fn=self.train_fn,
                validation_fn=self.validation_fn,
                experiment_name=self.name,
                previous_benchmark=previous_benchmark,
                DataManager=self.DataManager,
                name=name,
            )
            case_list.append(case_obj)
            self.complete = self.complete & case_obj.complete
            self.set_predict_score(case_obj.predict_score)
        write_dict_to_file(self.predict_score, self.predict_score_path)
        # # Iterate over the dictionary
        # for key, value in variables.items():
        #     # Check if the value is a list
        #     if isinstance(value, list):
        #         # If the value is a list, iterate over it
        #         for v in value:
        #             if isinstance(v, int):
        #                 case_name = v
        #             elif isinstance(v, str):
        #                 case_name=v
        #             else:
        #                 case_name = v.name
        #                 # Just the 1st letter of each word
        #                 case_name = "".join(
        #                     [f[0] for f in case_name.split("_")]
        #                 )
        #             # Create a Case object for each entry in the list
        #             case_obj = Case(
        #                 **{key: v},
        #                 **{k: v for k, v in variables.items() if k != key},
        #                 metrics=metrics,
        #                 callbacks=callbacks,
        #                 model_name=model_name,
        #                 case_name=case_name,
        #                 epochs=epochs,
        #                 model=model,
        #                 name=None,
        #                 work_folder=self.case_work_path,
        #                 train_fn=self.train_fn,
        #                 validation_fn=self.validation_fn,
        #                 experiment_name=self.name,
        #             )
        #             case_list.append(case_obj)
        #             self.complete = self.complete & case_obj.complete
        #             self.set_predict_score(case_obj.predict_score)
        if len(case_list) == 0:
            # If the value is not a list, create a Case object with the value
            case_obj = Case(
                **variables,
                metrics=metrics,
                callbacks=callbacks,
                epochs=epochs,
                model_name=model_name,
                model=model,
                name=None,
                work_folder=self.case_work_path,
                train_fn=self.train_fn,
                validation_fn=self.validation_fn,
                experiment_name=self.name,
                previous_benchmark=previous_benchmark,
                DataManager=self.DataManager,
            )
            case_list.append(case_obj)
            self.complete = self.complete & case_obj.complete
            self.set_predict_score(case_obj.predict_score)

        # if weight:
        #     if "delta_mean" in  weight or "both" in weight:
        #         train_dataset_Y_values = train_dataset_Y_values or train_dataset_Y
        #         samples_weights = np.abs(train_dataset_Y_values - mean)
        #         fit_args["sample_weight"] = samples_weights
        #     if "freq" in  weight or "both" in weight:
        #         freq_weights = get_freq_samples(train_dataset_labels)
        #         fit_args["sample_weight"] = freq_weights
        #     if "both" in weight:
        #         fit_args["sample_weight"] = freq_weights*samples_weights

        return case_list

    def experiment_configuration(self, models_dict=None, **kwargs):
        # TODO: REMOVE list in future and use only the dict base thingy, maybe change name
        models_dict = models_dict or self.models_dict
        self.conf = []
        self.study_cases = {}
        if self.previous_cases:
            for previous_case_obj in self.previous_cases:
                # if "UNET" in previous_case_obj.name:
                #     continue
                variables = {
                    "optimizer": isinstance(self.optimizer, list),
                    "loss": isinstance(self.loss, list),
                    "batch_size": isinstance(self.batch_size, list),
                }
                if variables["optimizer"]:
                    opt = self.optimizer
                else:
                    opt = previous_case_obj.optimizer

                if variables["loss"]:
                    lll = self.loss
                else:
                    lll = previous_case_obj.loss
                if variables["batch_size"]:
                    bbss = self.batch_size
                else:
                    bbss = previous_case_obj.batch_size
                # name I want previous_case_obj.name
                previous_benchmark = previous_case_obj.best_result
                if self.previous_experiment.best_result:
                    previous_benchmark = self.previous_experiment.best_result

                case_conf_list = self.case_configuration(
                    model_name=previous_case_obj.model_name,
                    model=previous_case_obj.model,
                    optimizer=opt,
                    loss=lll,
                    batch_size=bbss,
                    metrics=previous_case_obj.metrics,
                    epochs=self.epochs,
                    callbacks=self.callbacks,
                    # weight=previous_case_obj.weight
                    previous_benchmark=previous_benchmark,
                    name=previous_case_obj.name,
                    previous_case=previous_case_obj,
                    **kwargs,
                )
                for case_obj in case_conf_list:
                    self.conf.append(case_obj)
                    self.study_cases[case_obj.name] = case_obj
        else:
            for model_name, model in models_dict.items():
                case_conf_list = self.case_configuration(
                    model_name=model_name, model=model, **kwargs
                )
                for case_obj in case_conf_list:
                    self.conf.append(case_obj)
                    self.study_cases[case_obj.name] = case_obj
        return self.conf

    def validate_experiment(self):
        print("----------------------------------------------------------")
        print(f"Validating {self.name}")
        for case_obj in self.conf:
            self.set_predict_score(case_obj.predict_score)
            if not self.best_result:
                self.best_result = case_obj.best_result
            else:
                if case_obj.best_result:
                    self.best_result = max(
                        [case_obj.best_result, self.best_result]
                    )

        self.write_report()
        self.save(self.conf_file)

    # TODO: change all this mess of report writing
    # TODO: outsource this somewhere else, its too specific for my thesis case.
    def write_report(self):
        print(f"writin report for {self.name}")
        import pandas as pd

        # TODO: change this path thingys
        case_report_path = self.case_work_path.replace(
            "experiments", "reports"
        )
        os.makedirs(case_report_path, exist_ok=True)

        case_results = pd.DataFrame(self.predict_score)

        COLUMN_TO_SORT_BY = VALIDATION_TARGET
        ascending_to_sort = False

        #  Best result
        self.best_result = max(case_results[VALIDATION_TARGET])

        # Best from previous
        if self.previous_experiment:
            better_scores = case_results[
                case_results[VALIDATION_TARGET]
                >= self.previous_experiment.best_result
            ]
        else:
            if os.path.exists(self.benchmark_score_file):
                benchmark_score = load_json_dict(self.benchmark_score_file)
            # if bscore then better than abs benchmark
            if VALIDATION_TARGET in ["bscore", "bscoreB"]:
                better_scores = case_results[
                    case_results["abs error"] <= benchmark_score["abs error"]
                ]
                better_scores = better_scores[
                    better_scores[VALIDATION_TARGET] > 0
                ]

            # if bscore_norm then higher bscore_norm>0 when missing and surpulr better than benchmark
            elif VALIDATION_TARGET in ["bscore_norm", "bscore_normB"]:
                better_scores = case_results[
                    case_results["alloc missing"]
                    <= benchmark_score["alloc missing"]
                ]
                better_scores = better_scores[
                    better_scores["alloc surplus"]
                    <= benchmark_score["alloc surplus"]
                ]
        unique_values_list = better_scores["name"].unique().tolist()
        # just the best
        max_target = better_scores[VALIDATION_TARGET].max()
        rows_with_best = better_scores[
            better_scores[VALIDATION_TARGET] == max_target
        ]
        unique_values_list = rows_with_best["name"].unique().tolist()

        better_scores.reset_index(inplace=True, drop=True)
        if self.forward_epochs:
            forward_cases = case_results[
                case_results["name"].isin(unique_values_list)
            ]
            forward_cases = forward_cases[
                forward_cases["epoch"] <= self.forward_epochs
            ]
            self.best_result = max(forward_cases[VALIDATION_TARGET])

        print("----------------------------------------------------")
        print("Worthy models are: ", unique_values_list)
        self.worthy_models = unique_values_list
        self.worthy_cases = [
            self.study_cases[f]
            for f in unique_values_list
            # if self.study_cases[f].worthy
        ]
        if len(better_scores) > 0:
            unique_values_list = better_scores["name"].unique().tolist()
            path_schema_tex = os.path.join(
                case_report_path,
                f"experiment_results_{VALIDATION_TARGET}_better_than_previous_10.tex",
            )

            better_scores.head(10).to_latex(
                path_schema_tex, escape=False, index=False, float_format="%.2f"
            )

            path_schema_tex = os.path.join(
                case_report_path,
                f"experiment_results_{VALIDATION_TARGET}_better_than_previous.tex",
            )

            better_scores.to_latex(
                path_schema_tex, escape=False, index=False, float_format="%.2f"
            )
            # Get the best score for each unique value in the "name" column
            best_scores = better_scores.loc[
                better_scores.groupby("name").idxmax()[COLUMN_TO_SORT_BY]
            ].sort_values(by=COLUMN_TO_SORT_BY, ascending=ascending_to_sort)

            path_schema_tex = os.path.join(
                case_report_path,
                f"experiment_results_{VALIDATION_TARGET}_better_than_previous_best_of_each.tex",
            )

            best_scores.to_latex(
                path_schema_tex, escape=False, index=False, float_format="%.2f"
            )

        # All results, ordered by epoch
        path_schema_csv = os.path.join(
            case_report_path, f"experiment_results_{VALIDATION_TARGET}.csv"
        )
        path_schema_tex = os.path.join(
            case_report_path, f"experiment_results_{VALIDATION_TARGET}.tex"
        )
        case_results.sort_values(
            by=COLUMN_TO_SORT_BY, ascending=ascending_to_sort, inplace=True
        )
        if len(case_results) > 10:
            case_results_sort = pd.concat(
                [case_results.head(), case_results.tail()]
            )
            case_results_sort.to_csv(path_schema_csv, index=False)
            case_results_sort.to_latex(
                path_schema_tex, escape=False, index=False, float_format="%.2f"
            )
            path_schema_csv = os.path.join(
                case_report_path,
                f"experiment_results_{VALIDATION_TARGET}_complete.csv",
            )
            path_schema_tex = os.path.join(
                case_report_path,
                f"experiment_results_{VALIDATION_TARGET}_complete.tex",
            )

        case_results.to_csv(path_schema_csv, index=False)
        case_results.to_latex(
            path_schema_tex, escape=False, index=False, float_format="%.2f"
        )

        # Get the best score for each unique value in the "name" column
        best_scores = case_results.loc[
            case_results.groupby("name").idxmax()[COLUMN_TO_SORT_BY]
        ].sort_values(by=COLUMN_TO_SORT_BY, ascending=ascending_to_sort)

        path_schema_tex = os.path.join(
            case_report_path,
            f"experiment_results_{VALIDATION_TARGET}_best_of_each.tex",
        )

        best_scores.to_latex(
            path_schema_tex, escape=False, index=False, float_format="%.2f"
        )

        # Get the 2nd and 3rd best scores for each unique value in the "name" column
        if ascending_to_sort is False:
            second_third_best_scores = case_results.groupby("name").apply(
                lambda x: x.nlargest(3, COLUMN_TO_SORT_BY)
            )
        else:
            second_third_best_scores = case_results.groupby("name").apply(
                lambda x: x.nsmallest(3, COLUMN_TO_SORT_BY)
            )

        second_third_best_scores.sort_values(
            by=COLUMN_TO_SORT_BY, ascending=ascending_to_sort, inplace=True
        )
        path_schema_tex = os.path.join(
            case_report_path,
            f"experiment_results_{VALIDATION_TARGET}_best3.tex",
        )

        second_third_best_scores.to_latex(
            path_schema_tex, escape=False, index=False, float_format="%.2f"
        )

        no_missin_scores = case_results[case_results["bscore m"] >= 0]
        no_missin_scores = no_missin_scores[no_missin_scores["bscore s"] >= 0]
        no_missin_scores = no_missin_scores.dropna().sort_values(
            by=VALIDATION_TARGET, ascending=False
        )
        # if len(no_missin_scores[VALIDATION_TARGET]) > 0:
        #     self.best_result = max(no_missin_scores[VALIDATION_TARGET])

        path_schema_tex = os.path.join(
            case_report_path,
            f"experiment_results_{VALIDATION_TARGET}_best_10_under_benchmark.tex",
        )

        no_missin_scores.head(10).to_latex(
            path_schema_tex, escape=False, index=False, float_format="%.2f"
        )

        path_schema_tex = os.path.join(
            case_report_path,
            f"experiment_results_{VALIDATION_TARGET}_best_under_benchmark.tex",
        )

        no_missin_scores.to_latex(
            path_schema_tex, escape=False, index=False, float_format="%.2f"
        )
        if self.epochs > 50:
            no_missin_scores2 = case_results[case_results["bscore m"] >= 0]
            no_missin_scores2 = no_missin_scores2[
                no_missin_scores2["bscore s"] >= 0
            ]
            no_missin_scores2 = no_missin_scores2[
                no_missin_scores2["epoch"] <= 50
            ]

            no_missin_scores2 = no_missin_scores2.dropna().sort_values(
                by=VALIDATION_TARGET, ascending=False
            )
            # if len(no_missin_scores2[VALIDATION_TARGET]) > 0:
            #     self.best_result = max(no_missin_scores2[VALIDATION_TARGET])

            path_schema_tex = os.path.join(
                case_report_path,
                f"experiment_results_{VALIDATION_TARGET}_best_10_under_benchmarkminu50epochs.tex",
            )

            no_missin_scores2.head(10).to_latex(
                path_schema_tex, escape=False, index=False, float_format="%.2f"
            )

            path_schema_tex = os.path.join(
                case_report_path,
                f"experiment_results_{VALIDATION_TARGET}_best_under_benchmark_minu50epochs.tex",
            )

            no_missin_scores2.to_latex(
                path_schema_tex, escape=False, index=False, float_format="%.2f"
            )
        better_scores = []

        if self.previous_experiment:
            better_scores = no_missin_scores[
                no_missin_scores[VALIDATION_TARGET]
                >= self.previous_experiment.best_result
            ]
        if len(better_scores) > 0:
            unique_values_list = better_scores["name"].unique().tolist()
            path_schema_tex = os.path.join(
                case_report_path,
                f"experiment_results_{VALIDATION_TARGET}_better_than_previous_10_under_benchmark.tex",
            )

            better_scores.head(10).to_latex(
                path_schema_tex, escape=False, index=False, float_format="%.2f"
            )

            path_schema_tex = os.path.join(
                case_report_path,
                f"experiment_results_{VALIDATION_TARGET}_better_than_previous_under_benchmark.tex",
            )

            better_scores.to_latex(
                path_schema_tex, escape=False, index=False, float_format="%.2f"
            )
            # self.best_result = max(better_scores[VALIDATION_TARGET])

        else:
            unique_values_list = no_missin_scores["name"].unique().tolist()
        # print("----------------------------------------------------")
        # print("Worthy models are: ", unique_values_list)
        # self.worthy_models = unique_values_list
        # self.worthy_cases = [
        #     self.study_cases[f]
        #     for f in unique_values_list
        #     # if self.study_cases[f].worthy
        # ]

        if ascending_to_sort is False:
            no_missin_scores = no_missin_scores.groupby("name").apply(
                lambda x: x.nlargest(3, COLUMN_TO_SORT_BY)
            )
        else:
            no_missin_scores = no_missin_scores.groupby("name").apply(
                lambda x: x.nsmallest(3, COLUMN_TO_SORT_BY)
            )

        no_missin_scores.sort_values(
            by=VALIDATION_TARGET, ascending=False, inplace=True
        )

        path_schema_tex = os.path.join(
            case_report_path,
            f"experiment_results_{VALIDATION_TARGET}_best3_under_benchmark.tex",
        )

        no_missin_scores.to_latex(
            path_schema_tex, escape=False, index=False, float_format="%.2f"
        )

        # TODO: make a plot with the real data and the best predictions
        # the plot is the one year, one month, one week, one day
        # maybe the best and the worse of each
        benchmark_prediction_file = self.benchmark_score_file.replace(
            ".json", "npz"
        )

    def visualize_report(self):
        # TODO: change this path thingys
        folder_figures = self.case_work_path.replace("experiments", "reports")

        METRICS_TO_CHECK = os.getenv("METRICS_TO_CHECK", None)
        metrics_to_check = None
        if METRICS_TO_CHECK:
            metrics_to_check = METRICS_TO_CHECK.split("|")

        benchmark_score = {}

        if os.path.exists(self.benchmark_score_file):
            benchmark_score = load_json_dict(self.benchmark_score_file)

        figure_name = f"experiment_results_{VALIDATION_TARGET}.png"
        self.visualize_report_fn(
            self.predict_score,
            metrics_to_check=metrics_to_check,
            benchmark_score=benchmark_score,
            folder_figures=folder_figures,
            figure_name=figure_name,
        )

        self.visualize_report_fn(
            self.predict_score,
            metrics_to_check=metrics_to_check,
            benchmark_score=benchmark_score,
            folder_figures=folder_figures,
            figure_name=figure_name,
            limit_by="outliers",
        )
        self.visualize_report_fn(
            self.predict_score,
            metrics_to_check=metrics_to_check,
            benchmark_score=benchmark_score,
            folder_figures=folder_figures,
            figure_name=figure_name,
            limit_by="benchmark",
        )

        metrics_to_check = ["alloc missing", "alloc surplus"]
        figure_name = f"experiment_results_{VALIDATION_TARGET}_redux.png"
        figure_size = (20, 10)
        self.visualize_report_fn(
            self.predict_score,
            metrics_to_check=metrics_to_check,
            benchmark_score=benchmark_score,
            folder_figures=folder_figures,
            figure_name=figure_name,
            figsize=figure_size,
        )

        self.visualize_report_fn(
            self.predict_score,
            metrics_to_check=metrics_to_check,
            benchmark_score=benchmark_score,
            folder_figures=folder_figures,
            figure_name=figure_name,
            limit_by="outliers",
            figsize=figure_size,
        )
        self.visualize_report_fn(
            self.predict_score,
            metrics_to_check=metrics_to_check,
            benchmark_score=benchmark_score,
            folder_figures=folder_figures,
            figure_name=figure_name,
            limit_by="benchmark",
            figsize=figure_size,
        )
        metrics_to_check = None
        figure_name = f"experiment_results_{VALIDATION_TARGET}_case.png"
        if "case" in self.predict_score:
            self.visualize_report_fn(
                self.predict_score,
                metrics_to_check=metrics_to_check,
                benchmark_score=benchmark_score,
                folder_figures=folder_figures,
                figure_name=figure_name,
                column_to_group="case",
            )

            self.visualize_report_fn(
                self.predict_score,
                metrics_to_check=metrics_to_check,
                benchmark_score=benchmark_score,
                folder_figures=folder_figures,
                figure_name=figure_name,
                limit_by="outliers",
                column_to_group="case",
            )
            self.visualize_report_fn(
                self.predict_score,
                metrics_to_check=metrics_to_check,
                benchmark_score=benchmark_score,
                folder_figures=folder_figures,
                figure_name=figure_name,
                limit_by="benchmark",
                column_to_group="case",
            )


def ExperimentFactory(
    target_variable=None,
    previous_experiment_dict=None,
    name=None,
    get_models_on_experiment_fn=None,
    get_models_args=None,
    **kwargs,
):
    from data.definitions import ALL_DATA_MANAGERS

    target_variable = target_variable or os.getenv("TARGET_VARIABLE")
    # target_variable = "Upward;Downward"
    # target_variable = "Upward|Downward"
    # target_variable = "Upward;Downward|Tender"
    # target_variable = "Upward;Downward|Tender|Upward;Downward"
    # target_variable = "Upward;Downward|Upward;Downward"
    # target_variable = "Upward;Downward|Tender|Upward;Downward"
    final_targets = get_target_dict(target_variable)
    experiment_dict = {}
    for tag_name, targets in final_targets.items():
        previous_experiment = None
        exp_name = f"{tag_name}:{name}"
        if previous_experiment_dict is not None:
            # TODO: change this getting of the previous name
            last_arch_name = list(previous_experiment_dict.keys())[0].split(
                ":"
            )[-1]
            last_tag_name = f"{tag_name}:{last_arch_name}"
            previous_experiment = previous_experiment_dict[last_tag_name]

        dataman = ALL_DATA_MANAGERS[tag_name]
        exp = Experiment(
            target_name=tag_name,
            target_variables=targets,
            DataManager=dataman,
            previous_experiment=previous_experiment,
            name=exp_name,
            get_models_on_experiment_fn=get_models_on_experiment_fn,
            get_models_args=get_models_args,
            **kwargs,
        )
        experiment_dict[exp_name] = exp

    return experiment_dict
