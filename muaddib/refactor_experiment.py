# ShaiHulud
import copy
import importlib
import inspect
import itertools
import os

import alquitable
import numpy as np
from keras_core.losses import MeanSquaredError

from muaddib.models import CaseModel, ModelHalleck
from muaddib.shaihulud_utils import (
    get_mirror_weight_loss,
    get_target_dict,
    is_jsonable,
    load_json_dict,
    open_model,
    write_dict_to_file,
)


class SpiceEyes:
    def __init__(
        self,
        work_folder=None,
        name=None,
        DataManager=None,
        obj_type=None,
        target_name=None,
        target_variables=None,
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
        get_models_on_experiment_fn=None,
        get_models_args=None,
    ):
        self.work_folder = work_folder
        self.name = name
        if name is not None:
            folder_name = self.name.split(":")[-1]
        self.DataManager = DataManager

        self.obj_type = obj_type
        if conf_file is None:
            if self.obj_type == "experiment":
                self.obj_work_folder = os.path.join(
                    work_folder, DataManager.name, folder_name
                )

            elif self.obj_type == "case":
                self.obj_work_folder = os.path.join(work_folder, self.name)
            self.conf_file = os.path.join(
                self.obj_work_folder, f"{obj_type}_conf.json"
            )
        else:
            self.conf_file = conf_file
            self.obj_work_folder = os.path.dirname(conf_file)
        # If there is a conf file load it, otherwise, setup
        if os.path.exists(self.conf_file):
            self.__dict__ = self.load(self.conf_file)
            self.after_load_setup()

        else:
            self.epochs = epochs
            self.optimizer = optimizer
            self.batch_size = batch_size
            self.callbacks = callbacks or []
            self.loss = loss or MeanSquaredError()
            self.metrics = metrics or ["root_mean_squared_error"]
            self.train_fn = train_fn
            self.validation_fn = validation_fn
            self.visualize_report_fn = visualize_report_fn
            self.keras_backend = keras_backend
            self.complete = False
            self.predict_score = {}
            self.benchmark_score_file = benchmark_score_file
            self.worthy_models = None
            self.predict_score_path = None
            self.setup()

    def save(self, path=None):
        dict_to_load = self.__dict__.copy()
        dict_to_save = {}
        for key, value in dict_to_load.items():
            if key in ["fit_args", "compile_args"]:
                value = None
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
            if key == "model_obj":
                dict_to_save[key] = self.model_case_obj.conf_file

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
        print(dict_to_save)
        write_dict_to_file(dict_to_save, path)

    @staticmethod
    def load(path=None):
        dict_to_load = load_json_dict(path)
        # Create a new dictionary to store the loaded data
        dict_to_restore = {}

        # Restore the original data structures and objects
        for key, value in dict_to_load.items():
            dict_to_restore[key] = value
            if value is None:
                continue
            if key in ["fit_args", "compile_args"]:
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
            elif key == "model_obj":
                dict_to_restore[key] = open_model(value)
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
                        "experiment_conf.json",
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
            elif key == "model_case_obj":
                dict_to_restore[key] = CaseModel(name=value)
            elif key == "previous_experiment":
                folder_target, folder_exp = value.split(":")
                path_experiment = os.path.join(
                    os.getenv("EXPERIMENT_FOLDER"),
                    folder_target,
                    folder_exp,
                    "experiment_conf.json",
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

    def obj_setup(self):
        pass

    def after_load_setup(self):
        pass

    def setup(self):
        os.makedirs(self.obj_work_folder, exist_ok=True)
        self.predict_score_path = os.path.join(
            self.obj_work_folder, "predict_score.json"
        )

        self.obj_setup()
        self.save(path=self.conf_file)


class Case(SpiceEyes):
    def __init__(self, model_case_obj=None, **kwargs):
        self.model_case_obj = model_case_obj
        super().__init__(obj_type="case", **kwargs)

    def set_compile_args(self):
        compile_args = {
            "optimizer": self.optimizer,
            "loss": self.loss,
            "metrics": self.metrics,
        }

        self.compile_args = compile_args

    def set_fit_args(self):
        epocs_to_train = self.epochs - self.model_case_obj.last_epoch
        if epocs_to_train < self.epochs:
            self.some_training = True

        if epocs_to_train < 1:
            self.complete = True
            return

        callbacks = []
        if not isinstance(self.callbacks, list):
            self.callbacks = [self.callbacks]

        for callback in self.callbacks:
            callback_args = {}
            arg_names = inspect.getfullargspec(callback).args
            if "save_frequency" in arg_names:
                callback_args["save_frequency"] = 1
            if "start_epoch" in arg_names:
                callback_args["start_epoch"] = self.model_case_obj.last_epoch
            if "model_keras_filename" in arg_names:
                frq_model_filename_sof = (
                    f"{self.model_case_obj.freq_saves_path}" + "/{epoch}.keras"
                )
                callback_args["model_keras_filename"] = frq_model_filename_sof
            if "filepath" in arg_names:
                callback_args["filepath"] = self.model_keras_path
            if "model_log_filename" in arg_names:
                callback_args["model_log_filename"] = self.model_log_path
            if "logs" in arg_names:
                if os.path.exists(self.model_log_path):
                    history = load_json_dict(self.model_log_path)
                    callback_args["logs"] = history
            callbacks.append(callback(**callback_args))

        fit_args = {
            "epochs": epocs_to_train,
            "callbacks": callbacks,
            "batch_size": self.batch_size,
        }

        self.fit_args = fit_args

    def halleck_setup(self):
        self.model_case_obj.set_case_model(
            loss=self.loss, CASE_MODEL_FOLDER=self.obj_work_folder
        )
        self.model_obj = self.model_case_obj.model_obj

    def after_load_setup(self):
        self.obj_setup()

    def obj_setup(self):
        self.halleck_setup()
        self.model_keras_path = os.path.join(
            self.obj_work_folder, f"{self.name}.keras"
        )
        self.model_log_path = os.path.join(
            self.obj_work_folder, "case_log.json"
        )
        self.set_compile_args()
        self.set_fit_args()

    def train_model(self):
        if self.complete:
            return
        print("-----------------------------------------------------------")
        # print(f"Training Model {self.name} from {self.experiment_name}")

        self.train_fn(
            model_to_train=self.model_obj,
            datamanager=self.DataManager,
            fit_args=self.fit_args,
            compile_args=self.compile_args,
            model_name=self.name,
        )
        self.complete = True


class Experiment(SpiceEyes):
    def __init__(
        self,
        halleck_obj=None,
        experiment_archs=None,
        previous_experiment=None,
        forward_epochs=None,
        **kwargs,
    ):
        self.halleck_obj = halleck_obj
        self.experiment_archs = experiment_archs
        self.previous_experiment = previous_experiment
        self.forward_epochs = forward_epochs
        super().__init__(obj_type="experiment", **kwargs)

    def obj_setup(self, **kwargs):
        self.halleck_configuration()
        self.experiment_configuration()
        pass

    def halleck_configuration(self):
        halleck_conf_file = os.path.join(
            self.obj_work_folder, "halleck_conf.json"
        )
        previous_halleck = None
        if self.previous_experiment:
            previous_halleck = self.previous_experiment.halleck_obj

        if os.path.exists(halleck_conf_file):
            self.halleck_obj = ModelHalleck(conf_file=halleck_conf_file)
        if self.halleck_obj is None:
            halleck_init_args = {
                "X_timeseries": self.DataManager.X_timeseries,
                "Y_timeseries": self.DataManager.Y_timeseries,
                "n_features_predict": self.DataManager.n_features_predict,
                "n_features_train": self.DataManager.n_features_train,
                "conf_file": halleck_conf_file,
            }
            if self.experiment_archs:
                halleck_init_args.update(
                    {"archs_to_use": self.experiment_archs}
                )
            elif self.previous_experiment:
                halleck_init_args.update(
                    {"previous_halleck": previous_halleck}
                )

            self.halleck_obj = ModelHalleck(**halleck_init_args)
        self.halleck_obj.setup(conf_file=halleck_conf_file)

    def experiment_configuration(self):
        # configure hallecks
        list_vars = [
            "optimizer",
            "loss",
            "batch_size",
        ]
        refactor_combinations = {}
        for var in list_vars:
            self_var = getattr(self, var)
            if isinstance(self_var, list):
                refactor_combinations[var] = self_var
            else:
                var_to_use = None
                if self.previous_experiment:
                    var_to_use = getattr(self.previous_experiment, var)
                var_to_use = var_to_use or self_var
                refactor_combinations[var] = [var_to_use]

        # Generate all combinations of values
        combinations = list(itertools.product(*refactor_combinations.values()))

        # Create a new dictionary for each combination
        result_combinations = [
            dict(zip(refactor_combinations.keys(), combination))
            for combination in combinations
        ]
        self.conf = []
        self.study_cases = {}
        commun_case_args = {
            "work_folder": self.obj_work_folder,
            "DataManager": self.DataManager,
            "epochs": self.epochs,
            "callbacks": self.callbacks,
            "train_fn": self.train_fn,
        }
        # loop the different models and then loop on the diferent exp cases
        # diferent models are either from the prvious exp or from the hallec
        for (
            model_name,
            casemodelobj,
        ) in self.halleck_obj.models_to_experiment.items():
            for case_args in result_combinations:
                case_name = model_name
                for k, n in case_args.items():
                    if isinstance(n, int):
                        name_to_add = str(n)
                    elif isinstance(n, str):
                        name_to_add = n
                    else:
                        name_to_add = n.name
                        if k == "loss":
                            name_to_add = "".join(
                                [f[0] for f in name_to_add.split("_")]
                            )

                    case_name += f"_{name_to_add}"
                if case_name.endswith("_"):
                    case_name = case_name[:-1]

                casemodelobj_to_use = copy.deepcopy(casemodelobj)
                case_obj = Case(
                    model_case_obj=casemodelobj_to_use,
                    name=case_name,
                    **case_args,
                    **commun_case_args,
                )
                self.conf.append(case_obj)
                self.study_cases[case_obj.name] = case_obj

        return self.conf


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
    final_targets = get_target_dict(target_variable)
    experiment_dict = {}
    for tag_name in final_targets.keys():
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
            DataManager=dataman,
            previous_experiment=previous_experiment,
            name=exp_name,
            get_models_on_experiment_fn=get_models_on_experiment_fn,
            get_models_args=get_models_args,
            **kwargs,
        )
        experiment_dict[exp_name] = exp

    return experiment_dict
