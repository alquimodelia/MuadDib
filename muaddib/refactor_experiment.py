# ShaiHulud
import copy
import importlib
import inspect
import itertools
import os

import alquitable
import numpy as np
from keras.losses import MeanSquaredError

from muaddib.models import CaseModel, ModelHalleck
from muaddib.shaihulud_utils import (
    flatten_extend,
    get_mirror_weight_loss,
    get_target_dict,
    is_jsonable,
    load_json_dict,
    write_dict_to_file,
)

VALIDATION_TARGET = os.getenv("VALIDATION_TARGET", "EPEA")


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
        activation_end=None,
        activation_middle=None,
        model_types=".keras",
        final_experiment=False,
        setup_args=None,
        weights=False,
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
            # if "Transformer" in self.name:
            #     self.epochs = 10
            self.optimizer = optimizer
            self.batch_size = batch_size
            self.activation_end = activation_end
            self.activation_middle = activation_middle
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
            self.validation_complete = False
            self.model_types = model_types
            self.best_result = None
            self.final_experiment = final_experiment
            setup_args = setup_args or {}
            self.weights = weights
            self.setup(**setup_args)

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
            if key in ["previous_halleck", "halleck_obj"]:
                dict_to_save[key] = value.conf_file
            if not is_jsonable(dict_to_save[key]):
                dict_to_save[key] = (
                    dict_to_save[key].name
                    if hasattr(dict_to_save[key], "name")
                    else str(dict_to_save[key])
                )
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
                continue
                # dict_to_restore[key] = open_model(value)
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
            elif key in ["previous_halleck", "halleck_obj"]:
                from muaddib.models import ModelHalleck

                dict_to_restore[key] = ModelHalleck(conf_file=value)

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

    def setup(self, **kwargs):
        os.makedirs(self.obj_work_folder, exist_ok=True)
        self.predict_score_path = os.path.join(
            self.obj_work_folder, "predict_score.json"
        )

        setup = self.obj_setup(**kwargs)
        if setup:
            self.save(path=self.conf_file)


class Case(SpiceEyes):
    def __init__(self, model_case_obj=None, on_study_name=None, **kwargs):
        self.model_case_obj = model_case_obj
        self.on_study_name = on_study_name
        super().__init__(obj_type="case", **kwargs)

    def set_compile_args(self):
        compile_args = {
            "optimizer": self.optimizer,
            "loss": self.loss,
            "metrics": self.metrics,
        }

        self.compile_args = compile_args

    def set_fit_args(self, run_anyway=False):
        epocs_to_train = self.epochs - self.model_case_obj.last_epoch

        if epocs_to_train < self.epochs:
            self.some_training = True

        if epocs_to_train < 1:
            self.complete = True
            if not run_anyway:
                return False
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
        return True

    def halleck_setup(self):
        self.model_case_obj.set_case_model(
            loss=self.loss, CASE_MODEL_FOLDER=self.obj_work_folder
        )
        self.model_obj = self.model_case_obj.model_obj

    def after_load_setup(self):
        self.obj_setup()

    def obj_setup(self, exp_configure=True, **kwargs):
        self.halleck_setup()
        self.model_keras_path = os.path.join(
            self.obj_work_folder, f"{self.name}.keras"
        )
        self.model_log_path = os.path.join(
            self.obj_work_folder, "case_log.json"
        )
        self.set_compile_args()
        self.set_fit_args()
        return True

    def train_model(self, run_anyway=False):
        if self.complete and not run_anyway:
            return
        elif not getattr(self, "fit_args", False):
            self.set_compile_args()
            if not self.set_fit_args(run_anyway):
                return
        print("inside the traing get here")
        print(self.fit_args)
        print(self.compile_args)
        print(self.weights)

        print("-----------------------------------------------------------")
        # print(f"Training Model {self.name} from {self.experiment_name}")
        if self.fit_args["epochs"]>0:
            self.train_fn(
                model_to_train=self.model_obj,
                datamanager=self.DataManager,
                fit_args=self.fit_args,
                compile_args=self.compile_args,
                model_name=self.name,
                weights=self.weights,
            )
        self.complete = True

    def validate_model(self):
        REDO_VALIDATION = os.getenv("REDO_VALIDATION", False)
        if isinstance(REDO_VALIDATION, str):
            if REDO_VALIDATION.lower() == "false":
                REDO_VALIDATION = False
            else:
                REDO_VALIDATION = True
        prediction_score_exists = os.path.exists(self.predict_score_path)
        validation_complete = (
            self.validation_complete or prediction_score_exists
        )
        do_validation = not validation_complete or REDO_VALIDATION
        if do_validation:
            for freq_save in self.model_case_obj.list_freq_saves:
                # TODO: model_types from hallek or case
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
                do_freq_validation = not validation_done or REDO_VALIDATION
                if do_freq_validation:
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
                    if self.on_study_name:
                        predict_score["case"] = self.on_study_name
                self.set_predict_score(predict_score)
            # TODO: perfect example to just change to tinyDB
            write_dict_to_file(self.predict_score, self.predict_score_path)
        else:
            self.predict_score = load_json_dict(self.predict_score_path)
        self.validation_complete = True
        self.save(self.conf_file)


class Experiment(SpiceEyes):
    def __init__(
        self,
        halleck_obj=None,
        experiment_archs=None,
        previous_experiment=None,
        forward_epochs=None,
        backward_epochs=None,
        final_experiment_fn=None,
        mode_for_best=None,
        VALIDATION_TARGET_EXP=None,
        **kwargs,
    ):
        self.halleck_obj = halleck_obj
        self.experiment_archs = experiment_archs
        self.previous_experiment = previous_experiment
        self.forward_epochs = forward_epochs
        self.backward_epochs = backward_epochs
        self.final_experiment_fn = final_experiment_fn
        self.mode_for_best = mode_for_best or os.getenv("MODE_FOR_BEST", None)
        self.VALIDATION_TARGET_EXP=VALIDATION_TARGET_EXP
        super().__init__(obj_type="experiment", **kwargs)
        if self.epochs<50:
            self.mode_for_best = "hightest_stable"
        if self.final_experiment:
            self.mode_for_best = "highest"

    @staticmethod
    def add(ExperimentA, ExperimentB):
        dict_to_loadA = ExperimentA.__dict__.copy()
        dict_to_loadB = ExperimentB.__dict__.copy()

        all_keys = list(
            set(list(dict_to_loadA.keys()) + list(dict_to_loadB.keys()))
        )

        new_dict_Halleck = {}

        for key in all_keys:
            if key == "name":
                tag, nameA = dict_to_loadA[key].split(":")
                tag, nameB = dict_to_loadB[key].split(":")

                value_to_use = f"{tag}:{nameA}+{nameB}"
                new_dict_Halleck[key] = value_to_use
            elif key in ["previous_halleck", "halleck_obj"]:
                from muaddib.models import ModelHalleck

                value_to_use = ModelHalleck.add(
                    dict_to_loadB[key], dict_to_loadB[key]
                )
            elif key == "DataManager":
                # Assuming the sum is always with the same dataset
                value = dict_to_loadA[key]
                new_dict_Halleck[key] = value
            elif key == "best_result":
                A = dict_to_loadA[key] or np.nan
                B = dict_to_loadB[key] or np.nan
                value = np.nanmax([A, B])
                if np.isnan(value):
                    continue
                new_dict_Halleck[key] = value
            elif key == "epochs":
                A = dict_to_loadA[key]
                B = dict_to_loadB[key]
                value = min(A, B)
                new_dict_Halleck[key] = value

            else:
                value_to_use = None
                A = None
                B = None
                if key in dict_to_loadA:
                    A = dict_to_loadA[key]
                if key in dict_to_loadB:
                    B = dict_to_loadB[key]
                if A == B:
                    value_to_use = A
                elif not isinstance(A, dict):
                    if A is None:
                        value_to_use = B
                    elif B is None:
                        value_to_use = A
                    else:
                        if not isinstance(A, list):
                            A = [A]
                        A = list(set(A))
                        if not isinstance(B, list):
                            B = [B]
                        B = list(set(B))
                        if A == B:
                            value_to_use = A
                        else:
                            value_to_use = A + B
                        if len(value_to_use) == 0:
                            value_to_use = value_to_use[0]
                else:
                    models_to_experiment = {}
                    models_to_experiment.update(dict_to_loadA[key])
                    models_to_experiment.update(dict_to_loadB[key])
                    value_to_use = models_to_experiment

                new_dict_Halleck[key] = value_to_use

        list_pop = [
            "conf_file",
            "predict_score",
            "obj_work_folder",
            "obj_type",
            "predict_score_path",
        ]
        for t in list_pop:
            if t in new_dict_Halleck:
                new_dict_Halleck.pop(t)

        new_dict_Halleck["validation_complete"] = False
        new_dict_Halleck["complete"] = True

        new_dict_Halleck_initilize = new_dict_Halleck.copy()

        list_pop = [
            "study_cases",
            "conf",
            "worthy_models",
            "worthy_cases",
            "complete",
            "best_result",
            "validation_complete",
        ]

        for t in list_pop:
            if t in new_dict_Halleck_initilize:
                new_dict_Halleck_initilize.pop(t)

        new_Halleck = Experiment(**new_dict_Halleck_initilize)
        new_Halleck.__dict__.update(new_dict_Halleck)
        new_Halleck.setup(exp_configure=False)
        new_Halleck.__dict__.update(new_dict_Halleck)

        return new_Halleck

    def obj_setup(self, exp_configure=True, **kwargs):
        if self.previous_experiment:
            if not self.previous_experiment.validation_complete:
                return False
            if not getattr(
                self.previous_experiment.halleck_obj, "best_archs", False
            ):
                return False
        if exp_configure:
            self.halleck_configuration()
            self.experiment_configuration()
        self.set_benchmark_path()
        return True

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
                "activation_middle": self.activation_middle,
                "activation_end": self.activation_end,
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

    def get_case_obj(self, TARGET_VARIABLE_CASE_DICT, case_name, **kwargs):
        if case_name in list(TARGET_VARIABLE_CASE_DICT.keys()):
            if os.path.exists(TARGET_VARIABLE_CASE_DICT[case_name]):
                case_obj = Case(conf_file=TARGET_VARIABLE_CASE_DICT[case_name])
            else:
                case_obj = Case(**kwargs)
        else:
            case_obj = Case(**kwargs)
        return case_obj

    def experiment_configuration(self):
        TARGET_VARIABLE_CASE_DICT_PATH = os.path.join(os.path.dirname(self.obj_work_folder), "case_list_dict.json")
        TARGET_VARIABLE_CASE_DICT={}
        if os.path.exists(TARGET_VARIABLE_CASE_DICT_PATH):
            TARGET_VARIABLE_CASE_DICT = load_json_dict(TARGET_VARIABLE_CASE_DICT_PATH)
        # TODO: hacky way to just get MW or not
        if self.previous_experiment:
            if isinstance(self.loss, list):
                last_worthy_case_losses = [f.loss for f in self.previous_experiment.worthy_cases]
                last_worthy_case_losses = set(flatten_extend(last_worthy_case_losses))
                MW_losses = [
                    f for f in last_worthy_case_losses if "mirror" in f.name
                ]
                not_MW_losses = [
                    f
                    for f in last_worthy_case_losses
                    if "mirror" not in f.name
                ]
                if len(MW_losses) > 0:
                    self.loss = [f for f in self.loss if "mirror" in f.name]
                    advance_names_current = set([f.name.replace("mirror_", "").split("_")[0] for f in self.loss])
                    advance_names = [f.name.replace("mirror_", "").split("_")[0] for f in MW_losses]
                    # MW_losses_advance_prev=[f.name.replace("mirror_", "").split("_")[0] for f in MW_losses]
                    if len(advance_names_current)<=1:
                        self.loss = [f for f in self.loss if f.name.replace("mirror_", "").split("_")[0] in advance_names]
                    # self.loss = [f for f in self.loss if f.name in mirror_names]
                else:
                    self.loss = [
                        f for f in self.loss if "mirror" not in f.name
                    ]

        # TODO: keep track of what is being studied
        what_is_on_study = set()
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
                if len(self_var) == 1:
                    self_var = self_var[0]
            if isinstance(self_var, list):
                refactor_combinations[var] = self_var
                # TODO: keep track of what is being studied
                # experiment_variables.add(self_var)
                what_is_on_study.add(var)
            else:
                var_to_use = None
                # TODO: we should get this attributes from the best case and not the experiment itself. if is a list were fucked.
                # how do  we do for worthy worthy_cases? or for one best case scenrio?
                if self.previous_experiment:
                    var_to_use = getattr(self.previous_experiment, var)
                    if isinstance(var_to_use, list):
                        if len(var_to_use) == 1:
                            var_to_use = var_to_use[0]

                    if isinstance(var_to_use, list):
                        var_to_use = [
                            getattr(f, var)
                            for f in self.previous_experiment.worthy_cases
                        ]
                var_to_use = var_to_use or self_var
                setattr(self, var, var_to_use)
                if not isinstance(var_to_use, list):
                    var_to_use = [var_to_use]
                refactor_combinations[var] = var_to_use

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
            "validation_fn": self.validation_fn,
        }
        # loop the different models and then loop on the diferent exp cases
        # diferent models are either from the prvious exp or from the hallec
        for (
            model_name,
            casemodelobj,
        ) in self.halleck_obj.models_to_experiment.items():
            on_study_name = casemodelobj.case_to_study_name or ""
            for case_args in result_combinations:
                on_study_name_to_use = on_study_name
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
                    if k in what_is_on_study:
                        if len(on_study_name) == 0:
                            on_study_name_to_use = name_to_add
                        else:
                            on_study_name_to_use = (
                                on_study_name + f"_{name_to_add}"
                            )
                    case_name += f"_{name_to_add}"
                if case_name.endswith("_"):
                    case_name = case_name[:-1]

                casemodelobj_to_use = copy.deepcopy(casemodelobj)
                case_obj = None
                new_case_obj = None
                # TODO: Redo this conf of weigths cases, it seems that it will only work properly if exp1 not wieth and exp2 with
                if self.weights:
                    new_case_name = case_name + "_weights"
                    previous_weigths = False
                    if self.previous_experiment:
                        previous_weigths = self.previous_experiment.weights
                        if (
                            new_case_name
                            in self.previous_experiment.study_cases
                        ):
                            new_case_obj = (
                                self.previous_experiment.study_cases[
                                    new_case_name
                                ]
                            )
                    diference_wights_now_and_before = (
                        self.weights != previous_weigths
                    )
                    if new_case_obj is None:
                        if diference_wights_now_and_before:
                            new_case_obj = self.get_case_obj(TARGET_VARIABLE_CASE_DICT, new_case_name, model_case_obj=casemodelobj_to_use,
                                name=new_case_name,
                                on_study_name=on_study_name_to_use,
                                weights=True,
                                **case_args,
                                **commun_case_args,
                                )
                            # new_case_obj = Case(
                            #     model_case_obj=casemodelobj_to_use,
                            #     name=new_case_name,
                            #     on_study_name=on_study_name_to_use,
                            #     weights=True,
                            #     **case_args,
                            #     **commun_case_args,
                            # )
                    if new_case_obj:
                        self.conf.append(new_case_obj)
                        self.study_cases[new_case_obj.name] = new_case_obj
                if self.previous_experiment:
                    if case_name in self.previous_experiment.study_cases:
                        case_obj = self.previous_experiment.study_cases[
                            case_name
                        ]
                if case_obj is None:
                    case_obj = self.get_case_obj(TARGET_VARIABLE_CASE_DICT, case_name,model_case_obj=casemodelobj_to_use,
                        name=case_name,
                        on_study_name=on_study_name_to_use,
                        **case_args,
                        **commun_case_args,)
                    # case_obj = Case(
                    #     model_case_obj=casemodelobj_to_use,
                    #     name=case_name,
                    #     on_study_name=on_study_name_to_use,
                    #     **case_args,
                    #     **commun_case_args,
                    # )
                self.conf.append(case_obj)
                self.study_cases[case_obj.name] = case_obj
                if case_obj.name not in TARGET_VARIABLE_CASE_DICT:
                    TARGET_VARIABLE_CASE_DICT[case_obj.name]=case_obj.conf_file
        write_dict_to_file(TARGET_VARIABLE_CASE_DICT,TARGET_VARIABLE_CASE_DICT_PATH)
        return self.conf

    def validate_experiment(self):
        print("----------------------------------------------------------")
        print(f"Validating {self.name}")
        prediction_score_exists = os.path.exists(self.predict_score_path)
        REDO_VALIDATION = os.getenv("REDO_VALIDATION", False)
        if isinstance(REDO_VALIDATION, str):
            if REDO_VALIDATION.lower() == "false":
                REDO_VALIDATION = False
            else:
                REDO_VALIDATION = True
        validation_complete = (
            self.validation_complete or prediction_score_exists
        )
        do_validation = not validation_complete or REDO_VALIDATION
        if do_validation:
            for case_obj in self.conf:
                self.set_predict_score(case_obj.predict_score)
                if not self.best_result:
                    self.best_result = case_obj.best_result
                else:
                    if case_obj.best_result:
                        self.best_result = max(
                            [case_obj.best_result, self.best_result]
                        )

            self.validation_complete = True
            self.save(self.conf_file)
            write_dict_to_file(self.predict_score, self.predict_score_path)

        elif prediction_score_exists:
            self.predict_score = load_json_dict(self.predict_score_path)
        self.validation_complete = True
        self.write_report()
        # TODO: tinyDBBBB
        self.halleck_obj.set_best_case_model(self.worthy_cases)
        self.halleck_obj.save(self.halleck_obj.conf_file)

    # TODO: change all this mess of report writing
    # TODO: outsource this somewhere else, its too specific for my thesis case.
    # TODO: modulate this between what defnies new stuf and what saves figs and stuf
    def write_report(self, make_tex=True):
        import pandas as pd

        TARGET_VARIABLE_RESULTS_PATH = os.path.join(os.path.dirname(self.obj_work_folder), "results_score.csv")
        if os.path.exists(TARGET_VARIABLE_RESULTS_PATH):
            target_result_score = pd.read_csv(TARGET_VARIABLE_RESULTS_PATH,index_col=0)
        else:
            target_result_score = pd.DataFrame()
        print(f"writin report for {self.name}")

        # TODO: change this path thingys
        case_report_path = self.obj_work_folder.replace(
            "experiments", "reports"
        )
        os.makedirs(case_report_path, exist_ok=True)
        print("shame on you")
        print("self.VALIDATION_TARGET_EXP",self.VALIDATION_TARGET_EXP)
        print("VALIDATION_TARGET",VALIDATION_TARGET)

        COLUMN_TO_SORT_BY =  self.VALIDATION_TARGET_EXP or VALIDATION_TARGET
        print("COLUMN_TO_SORT_BY",COLUMN_TO_SORT_BY)

        ascending_to_sort = False

        case_results = pd.DataFrame(self.predict_score)

        case_results = case_results.drop_duplicates(["name", "epoch"])
        case_results = case_results.sort_values(["name", "epoch"])
        bbb = max(case_results[COLUMN_TO_SORT_BY])
        bbb_case = case_results[case_results[COLUMN_TO_SORT_BY] == bbb]
        unique_values_list = bbb_case["name"].unique().tolist()
        num = 1 if len(target_result_score)==0 else max(target_result_score["number"])+1
        row_dict = {"number":[num], "name":[self.name],"best_raw_case":unique_values_list, "COLUMN_TO_SORT_BY":[COLUMN_TO_SORT_BY],
                    "best_raw_value":[bbb] }


        print("best Value is:", bbb)
        print("Best case is:")
        print(bbb_case[["name", COLUMN_TO_SORT_BY, "epoch"]])
        if self.mode_for_best == "highest_stable":
            grouped_mean = (
                case_results[["name", COLUMN_TO_SORT_BY]]
                .groupby("name")
                .mean()
            )
            grouped_std = (
                case_results[["name", COLUMN_TO_SORT_BY]].groupby("name").std()
            )
            mean_mean = grouped_mean[COLUMN_TO_SORT_BY].mean()
            above_mean = grouped_mean[
                grouped_mean[COLUMN_TO_SORT_BY] >= mean_mean
            ]
            above_mean_name = above_mean.index.unique().tolist()
            above_mean_cases = case_results[
                case_results["name"].isin(above_mean_name)
            ]
            above_mean_cases_std = grouped_std[
                grouped_std.index.isin(above_mean_name)
            ]
            min_std = above_mean_cases_std.min().item()
            min_std_cases = above_mean_cases_std[
                above_mean_cases_std[COLUMN_TO_SORT_BY] <= min_std
            ]
            unique_values_list = min_std_cases.index.unique().tolist()
            case_results = case_results[
                case_results["name"].isin(unique_values_list)
            ]
        elif self.mode_for_best=="percentage_change":
            # WILL no WORK FOR NON LINEAR!
            case_results["pct"]=case_results.sort_values("epoch").groupby("name")[COLUMN_TO_SORT_BY].pct_change()
            case_results["acc"]=case_results.sort_values("epoch").groupby("name")["pct"].pct_change()
            max_epoch = self.epochs - max(int(0.2*self.epochs), 5)
            max_mean = case_results[case_results["epoch"]>=max_epoch].sort_values("epoch").groupby("name")["acc"].mean()


            max_mean = max(max_mean)
        elif self.mode_for_best=="polyfit":
            MAX_EPOCS=max(self.epochs, 200)
            case_results["wtf"] = None

            import copy

            from scipy.optimize import curve_fit
            from sklearn.metrics import (
                d2_absolute_error_score,
                mean_squared_error,
                r2_score,
            )

            from muaddib.stats_funcs import (
                ALL_MODELS_DICT_FUNCTION,
                get_mean_from_rolling,
            )
            case_results = case_results.drop_duplicates(["name", "epoch"])
            sorted_cases = case_results.sort_values(["name", "epoch"])
            # sorted_cases.loc[sorted_cases[COLUMN_TO_SORT_BY]<-100, COLUMN_TO_SORT_BY]=-100
            grouped = sorted_cases.groupby("name")
            

            import matplotlib.pyplot as plt
            figsize=(40,30)
            fig, axes = plt.subplots(nrows=len(case_results["name"].unique()), ncols=1, figsize=figsize
            )

            list_of_labels = len(ALL_MODELS_DICT_FUNCTION)+1
            # Get the number of unique labels
            num_labels = list_of_labels

            # Create a colormap with the desired number of colors
            colormap = plt.cm.get_cmap("viridis", num_labels)

            # Get a list of colors from the colormap
            handles = []
            labels = []

            cases_results_poly= {}
            i = 0
            for name, group in grouped:
                group_to_use = group.sort_values("epoch")
                x = group_to_use["epoch"]
                y = group_to_use[COLUMN_TO_SORT_BY]
                # y_min = np.abs(np.min(y))
                # # Subtract the minimum value from your data and add 1
                # y_shifted = y + y_min + 1
                # y_shifted_max = np.max(y_shifted)

                # # Apply logarithmic transformation to your data
                # y_log = np.log(y_shifted)

                # # Multiply the result by the desired maximum value divided by the new minimum value
                # y_scaled = y_log #* (100 + y_min) / y_shifted_max
                # # print(y)
                model_for_group = None
                model_for_group_name=None

                best_r2_result = None
                best_model_results = None
                roll_mean = None
                best_rmse =None
                best_met=None


                ax = np.array(axes).flatten()[i]
                import math
                win = int(math.ceil(len(group_to_use.set_index("epoch").sort_index()[COLUMN_TO_SORT_BY])*0.1))
                rrrrrr = get_mean_from_rolling(copy.deepcopy(group_to_use.set_index("epoch").sort_index()[COLUMN_TO_SORT_BY]), win)

                group_to_use.set_index("epoch").sort_index()[COLUMN_TO_SORT_BY].plot(ax=ax)
                ax.axhline(y=group_to_use.set_index("epoch").sort_index()[COLUMN_TO_SORT_BY].max(),color="blue", linestyle="--", label="best_vale")      
                ax.axhline(y=rrrrrr,color="orange", linestyle="dashdot", label="best_vale")      

                lines_after = np.array(axes).flatten()[i].get_lines()
                new_lines = [line for line in lines_after if line.get_label() not in labels]
                new_labels = [f.get_label() for f in lines_after if f.get_label() not in labels]
                handles += new_lines
                labels += new_labels
                # df_to_check = case_results[case_results["name"]==name][["name",COLUMN_TO_SORT_BY, "epoch"]]
                # df_to_check = df_to_check.sort_values(["name", "epoch"])

                case_plot = pd.DataFrame()
                # if len(y)>=MAX_EPOCS:
                #     cases_with_same_epochs_as_max = y[:MAX_EPOCS]
                #     cases_results_poly[name] = max(cases_with_same_epochs_as_max)
                #     continue

                for model_name in ALL_MODELS_DICT_FUNCTION.keys():




                    model = ALL_MODELS_DICT_FUNCTION[model_name]["model"]
                    p0 = ALL_MODELS_DICT_FUNCTION[model_name]["po"]
                    try:
                        popt, pcov = curve_fit(model, copy.deepcopy(x), copy.deepcopy(y), p0=p0)
                    except Exception as e:
                        print("Could not fit curv in ", model_name)
                        print(e)
                        continue
                    # best fit
                    model_results = [model(f, *popt) for f in range(1, max(MAX_EPOCS, len(y))+1)]
                    if np.sum(model_results==100)>len(model_results)*0.65:
                        continue
                    # model_results = [np.exp(f)-y_min-1 for f in model_results]

                    # model_results = [np.exp((f*y_shifted_max)/(100 + y_min))-y_min-1 for f in model_results]
                    if "epoch" not in case_plot:
                        case_plot["epoch"] = np.arange(len(model_results))+1

                    # print("ritititi")
                    # print(len(y))
                    # print(len(model_results))
                    # print(y)
                    # print(model_results)

                    r2_res = r2_score(y, model_results[:len(y)])
                    rmse = mean_squared_error(y, model_results[:len(y)],squared=False)

                    if "Downward" in TARGET_VARIABLE_RESULTS_PATH:
                        mdae=r2_res#d2_absolute_error_score(y, model_results[:len(y)])
                    else:
                        mdae=d2_absolute_error_score(y, model_results[:len(y)])

                    # print(r2_res)
                    # print(rmse)

                    # print("model_name", model_name)
                    # print("r2_res", r2_res)
                    # print("max", max(model_results))
                    # print("min", min(model_results))


                    case_plot[model_name] = model_results

                    # case_results.loc[case_results["name"]==name, model_name]=model_results[:len(case_results[case_results["name"]==name])]
                    # case_results.loc[case_results["name"]==name].set_index("epoch").sort_index()[model_name].plot(ax=ax)
                    import math
                    win = int(math.ceil(len(model_results)*0.1))
                    import copy
                    data_to_roll = copy.deepcopy(case_plot[model_name])
                    roll_mean_small = get_mean_from_rolling(data_to_roll, win)

                    # # since everything is way off 100, lets just skip the one near that
                    # if abs(roll_mean_small-100)<5:
                    #     roll_mean_small=-100
                    #     mdae=-100

                    
                    if roll_mean_small >0:
                        case_plot.set_index("epoch").sort_index()[model_name].plot(ax=ax)
                        color = np.array(axes).flatten()[i].get_lines()[-1].get_color()

                        # mdae = abs(roll_mean_small-rrrrrr)
                        # print("ooooooooooooooo")
                        # print(model_name)
                        # print(roll_mean_small)
                        # print(mdae)
                        np.array(axes).flatten()[i].axhline(y=roll_mean_small,color=color, linestyle="--", label=model_name+"_mean")      
                        # case_results.set_index("epoch").sort_index().groupby("name")[model_name].plot(ax=ax)

                        lines_after = np.array(axes).flatten()[i].get_lines()
                        new_lines = [line for line in lines_after if line.get_label() not in labels]
                        new_labels = [f.get_label() for f in lines_after if f.get_label() not in labels]
                        handles += new_lines
                        labels += new_labels

              


                    # df_to_check[model_name]=model_results[:len(case_results[case_results["name"]==name])]
                    if model_for_group is None:
                        model_for_group = model
                        best_r2_result = r2_res
                        best_model_results = model_results
                        model_for_group_name=model_name
                        roll_mean = roll_mean_small
                        best_rmse = rmse
                        best_met=mdae

                    if mdae>best_met:#rmse<best_rmse:#
                        best_r2_result = r2_res
                        model_for_group = model
                        best_model_results = model_results
                        model_for_group_name=model_name
                        roll_mean = roll_mean_small
                        best_rmse = rmse
                        best_met=mdae

                np.array(axes).flatten()[i].set_ylim([-100, 110])
                np.array(axes).flatten()[i].set_title(name)

                import math
                win = int(math.ceil(len(best_model_results)*0.1))
                cases_results_poly[name] = roll_mean
                np.array(axes).flatten()[i].axhline(y=roll_mean,color="black", linestyle="dashdot", label="winner")   
                i+=1

                # folder_figures = self.obj_work_folder.replace("experiments", "reports")

                # figure_name = f"case_results_{COLUMN_TO_SORT_BY}_{name}.png"



                case_results.loc[case_results["name"]==name, "wtf"]=best_model_results[:len(case_results[case_results["name"]==name])]

            folder_figures = self.obj_work_folder.replace("experiments", "reports")

            figure_name = f"case_results_{COLUMN_TO_SORT_BY}_2.png"
            fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))
            # plt.tight_layout()
            plt.savefig(os.path.join(folder_figures, figure_name),  bbox_inches="tight", pad_inches=0.1)
            plt.close()
            # print("............................................................")
            # print(cases_results_poly)
            max_key = max(cases_results_poly, key=cases_results_poly.get)
            max_value = cases_results_poly[max_key]
            # print("waht is wahr", max_key)
            # print("waht is value", max_value)
            # print("r2", best_r2_result)
            # print("waht is value", model_for_group_name)

            unique_values_list = [max_key]
            # print(case_results["name"].value_counts())
            case_results = case_results[
                case_results["name"].isin(unique_values_list)
            ]
            # print("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
            # print(case_results[["name", COLUMN_TO_SORT_BY, "wtf"]])



        if self.epochs < max(case_results["epoch"]):
            smaller_epochs = min(self.epochs, max(case_results["epoch"]))
            case_results = case_results[
                case_results["epoch"] <= smaller_epochs
            ]

        #  Best result
        self.best_result = max(case_results[COLUMN_TO_SORT_BY])
        if self.final_experiment:
            best_result_m = case_results[
                case_results[COLUMN_TO_SORT_BY] == self.best_result
            ]
            epoca = int(best_result_m["epoch"].item())
            best_result_m_name = best_result_m["name"].unique().tolist()[0]
            best_study_case = self.study_cases[best_result_m_name]
            prediciotn_saves = [
                f
                for f in best_study_case.model_case_obj.list_freq_saves
                if str(epoca) in f
            ]
            prediciotn_saves = [
                f.replace("freq_saves", "freq_predictions").replace(
                    ".keras", ".npz"
                )
                for f in prediciotn_saves
            ][0]
            if self.final_experiment_fn:
                self.final_experiment_fn(
                    prediciotn_saves,
                    f"{self.DataManager.name}:{best_result_m_name}",
                    COLUMN_TO_SORT_BY,
                )

        # Best from previous
        if self.previous_experiment:
            # print("get hehehehe")
            # print(max(case_results[COLUMN_TO_SORT_BY]))
            # print(self.previous_experiment.best_result)

            print("--------------------------------------------------------------------")
            print("self.previous_experiment.best_result", self.previous_experiment.best_result)
            if len(list(case_results["name"].unique()))==1:
                better_scores = case_results
            else:
                better_scores = case_results[
                    case_results[COLUMN_TO_SORT_BY]
                    >= self.previous_experiment.best_result
                ]

        else:
            if os.path.exists(self.benchmark_score_file):
                benchmark_score = load_json_dict(self.benchmark_score_file)

            better_scores = case_results[case_results[COLUMN_TO_SORT_BY] > 0]
            if len(better_scores) == 0:
                # if EPEA then better than abs benchmark
                if COLUMN_TO_SORT_BY in ["EPEA", "EPEA_Bench"]:
                    better_scores = case_results[
                        case_results["abs error"]
                        <= benchmark_score["abs error"]
                    ]
                    better_scores = better_scores[
                        better_scores[COLUMN_TO_SORT_BY] > 0
                    ]

                # if EPEA_norm then higher EPEA_norm>0 when missing and surpulr better than benchmark
                elif COLUMN_TO_SORT_BY in ["EPEA_norm", "EPEA_Bench_norm"]:
                    better_scores = case_results[
                        case_results["alloc missing"]
                        <= benchmark_score["alloc missing"]
                    ]
                    better_scores = better_scores[
                        better_scores["alloc surplus"]
                        <= benchmark_score["alloc surplus"]
                    ]
            if len(better_scores) == 0:
                # get the best 5%
                top_values = case_results[COLUMN_TO_SORT_BY].quantile(0.94)
                better_scores = case_results[
                    case_results[COLUMN_TO_SORT_BY] > top_values
                ]
            if len(better_scores) == 0:
                top_values_m = case_results["alloc missing"].quantile(0.60)
                top_values_s = case_results["alloc surplus"].quantile(0.60)

                better_scores = case_results[
                    case_results["alloc missing"] > top_values_m
                ]

                better_scores = better_scores[
                    better_scores["alloc surplus"] > top_values_s
                ]
        unique_values_list = better_scores["name"].unique().tolist()
        # just the best
        max_target = better_scores[COLUMN_TO_SORT_BY].max()

        rows_with_best = better_scores[
            better_scores[COLUMN_TO_SORT_BY] == max_target
        ]
        unique_values_list = rows_with_best["name"].unique().tolist()

        better_scores.reset_index(inplace=True, drop=True)
        # if self.backward_epochs:
        #     if self.backward_epochs!=self.epochs:
        #         smaller_epochs = min([self.backward_epochs, self.epochs])

        #         backward_cases = case_results[case_results["epoch"] <= smaller_epochs]
        #         self.best_result = max(backward_cases[COLUMN_TO_SORT_BY])
        #         unique_values_list = rows_with_best["name"].unique().tolist()

        #         backward_cases = backward_cases[
        #             backward_cases["epoch"] <= smaller_epochs
        #         ]
        if self.forward_epochs:
            forward_cases = case_results[
                case_results["name"].isin(unique_values_list)
            ]

            forward_cases = forward_cases[
                forward_cases["epoch"] <= self.forward_epochs
            ]
            self.best_result = max(forward_cases[COLUMN_TO_SORT_BY])

        print("----------------------------------------------------")
        print("Worthy models are: ", unique_values_list)
        self.worthy_models = unique_values_list
        self.worthy_cases = [
            self.study_cases[f]
            for f in unique_values_list
            # if self.study_cases[f].worthy
        ]
        self.save(self.conf_file)
        if make_tex:
            if len(better_scores) > 0:
                unique_values_list = better_scores["name"].unique().tolist()
                path_schema_tex = os.path.join(
                    case_report_path,
                    f"experiment_results_{COLUMN_TO_SORT_BY}_better_than_previous_10.tex",
                )

                better_scores.head(10).to_latex(
                    path_schema_tex,
                    escape=False,
                    index=False,
                    float_format="%.2f",
                )

                path_schema_tex = os.path.join(
                    case_report_path,
                    f"experiment_results_{COLUMN_TO_SORT_BY}_better_than_previous.tex",
                )

                better_scores.to_latex(
                    path_schema_tex,
                    escape=False,
                    index=False,
                    float_format="%.2f",
                )
                # Get the best score for each unique value in the "name" column
                best_scores = better_scores.loc[
                    better_scores.groupby("name").idxmax()[COLUMN_TO_SORT_BY]
                ].sort_values(
                    by=COLUMN_TO_SORT_BY, ascending=ascending_to_sort
                )

                path_schema_tex = os.path.join(
                    case_report_path,
                    f"experiment_results_{COLUMN_TO_SORT_BY}_better_than_previous_best_of_each.tex",
                )

                best_scores.to_latex(
                    path_schema_tex,
                    escape=False,
                    index=False,
                    float_format="%.2f",
                )

            # All results, ordered by epoch
            path_schema_csv = os.path.join(
                case_report_path, f"experiment_results_{COLUMN_TO_SORT_BY}.csv"
            )
            path_schema_tex = os.path.join(
                case_report_path, f"experiment_results_{COLUMN_TO_SORT_BY}.tex"
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
                    path_schema_tex,
                    escape=False,
                    index=False,
                    float_format="%.2f",
                )
                path_schema_csv = os.path.join(
                    case_report_path,
                    f"experiment_results_{COLUMN_TO_SORT_BY}_complete.csv",
                )
                path_schema_tex = os.path.join(
                    case_report_path,
                    f"experiment_results_{COLUMN_TO_SORT_BY}_complete.tex",
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
                f"experiment_results_{COLUMN_TO_SORT_BY}_best_of_each.tex",
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
                f"experiment_results_{COLUMN_TO_SORT_BY}_best3.tex",
            )

            second_third_best_scores.to_latex(
                path_schema_tex, escape=False, index=False, float_format="%.2f"
            )

            no_missin_scores = case_results[case_results["EPEA_F"] >= 0]
            no_missin_scores = no_missin_scores[
                no_missin_scores["EPEA_D"] >= 0
            ]
            no_missin_scores = no_missin_scores.dropna().sort_values(
                by=COLUMN_TO_SORT_BY, ascending=False
            )
            # if len(no_missin_scores[COLUMN_TO_SORT_BY]) > 0:
            #     self.best_result = max(no_missin_scores[COLUMN_TO_SORT_BY])

            path_schema_tex = os.path.join(
                case_report_path,
                f"experiment_results_{COLUMN_TO_SORT_BY}_best_10_under_benchmark.tex",
            )

            no_missin_scores.head(10).to_latex(
                path_schema_tex, escape=False, index=False, float_format="%.2f"
            )

            path_schema_tex = os.path.join(
                case_report_path,
                f"experiment_results_{COLUMN_TO_SORT_BY}_best_under_benchmark.tex",
            )

            no_missin_scores.to_latex(
                path_schema_tex, escape=False, index=False, float_format="%.2f"
            )
            if self.epochs > 50:
                no_missin_scores2 = case_results[case_results["EPEA_F"] >= 0]
                no_missin_scores2 = no_missin_scores2[
                    no_missin_scores2["EPEA_D"] >= 0
                ]
                no_missin_scores2 = no_missin_scores2[
                    no_missin_scores2["epoch"] <= 50
                ]

                no_missin_scores2 = no_missin_scores2.dropna().sort_values(
                    by=COLUMN_TO_SORT_BY, ascending=False
                )
                # if len(no_missin_scores2[COLUMN_TO_SORT_BY]) > 0:
                #     self.best_result = max(no_missin_scores2[COLUMN_TO_SORT_BY])

                path_schema_tex = os.path.join(
                    case_report_path,
                    f"experiment_results_{COLUMN_TO_SORT_BY}_best_10_under_benchmarkminu50epochs.tex",
                )

                no_missin_scores2.head(10).to_latex(
                    path_schema_tex,
                    escape=False,
                    index=False,
                    float_format="%.2f",
                )

                path_schema_tex = os.path.join(
                    case_report_path,
                    f"experiment_results_{COLUMN_TO_SORT_BY}_best_under_benchmark_minu50epochs.tex",
                )

                no_missin_scores2.to_latex(
                    path_schema_tex,
                    escape=False,
                    index=False,
                    float_format="%.2f",
                )
            better_scores = []

            if self.previous_experiment:
                better_scores = no_missin_scores[
                    no_missin_scores[COLUMN_TO_SORT_BY]
                    >= self.previous_experiment.best_result
                ]
            if len(better_scores) > 0:
                unique_values_list = better_scores["name"].unique().tolist()
                path_schema_tex = os.path.join(
                    case_report_path,
                    f"experiment_results_{COLUMN_TO_SORT_BY}_better_than_previous_10_under_benchmark.tex",
                )

                better_scores.head(10).to_latex(
                    path_schema_tex,
                    escape=False,
                    index=False,
                    float_format="%.2f",
                )

                path_schema_tex = os.path.join(
                    case_report_path,
                    f"experiment_results_{COLUMN_TO_SORT_BY}_better_than_previous_under_benchmark.tex",
                )

                better_scores.to_latex(
                    path_schema_tex,
                    escape=False,
                    index=False,
                    float_format="%.2f",
                )
                # self.best_result = max(better_scores[COLUMN_TO_SORT_BY])

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
                by=COLUMN_TO_SORT_BY, ascending=False, inplace=True
            )

            path_schema_tex = os.path.join(
                case_report_path,
                f"experiment_results_{COLUMN_TO_SORT_BY}_best3_under_benchmark.tex",
            )

            no_missin_scores.to_latex(
                path_schema_tex, escape=False, index=False, float_format="%.2f"
            )

            # TODO: make a plot with the real data and the best predictions
            # the plot is the one year, one month, one week, one day
            # maybe the best and the worse of each
            self.benchmark_prediction_file = self.benchmark_score_file.replace(
                ".json", "npz"
            )
        row_dict["best_result"]=[self.best_result]
        row_dict["worthy_model"]=self.worthy_models
        if len(target_result_score)>0:
            if row_dict["name"][0] in target_result_score["name"]:
                return
        # target_result_score = pd.concat([target_result_score,pd.DataFrame(row_dict)])
        # target_result_score.sort_values("number").to_csv(TARGET_VARIABLE_RESULTS_PATH)

    def visualize_report(self):
        # TODO: change this path thingys
        folder_figures = self.obj_work_folder.replace("experiments", "reports")

        # qry = "{folder_figures}/**/**.png"
        # if len(glob.glob(qry, recursive=True)) > 0:
        #     return
        COLUMN_TO_SORT_BY =  self.VALIDATION_TARGET_EXP or VALIDATION_TARGET

        METRICS_TO_CHECK = os.getenv("METRICS_TO_CHECK", None)
        metrics_to_check = None
        if METRICS_TO_CHECK:
            metrics_to_check = METRICS_TO_CHECK.split("|")

        benchmark_score = {}

        if os.path.exists(self.benchmark_score_file):
            benchmark_score = load_json_dict(self.benchmark_score_file)

        figure_name = f"experiment_results_{COLUMN_TO_SORT_BY}.png"
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
        figure_name = f"experiment_results_{COLUMN_TO_SORT_BY}_redux.png"
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
        figure_name = f"experiment_results_{COLUMN_TO_SORT_BY}_case.png"
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


def SumExperiments(experiment_dictA, experiment_dictB):
    tagsA = [f.split(":")[0] for f in experiment_dictA.keys()]
    tagsB = [f.split(":")[0] for f in experiment_dictB.keys()]
    unique_tags = list(set(tagsA + tagsB))

    experiment_dict = {}

    for tag in unique_tags:
        A = None
        B = None
        if tag in tagsA:
            A = [f for k, f in experiment_dictA.items() if k.startswith(tag)][
                0
            ]
        if tag in tagsB:
            B = [f for k, f in experiment_dictB.items() if k.startswith(tag)][
                0
            ]
        value_to_use = None
        if B is not None:
            if A is not None:
                value_to_use = Experiment.add(A, B)
            else:
                value_to_use = B
        if value_to_use is None:
            if A is not None:
                value_to_use = A
        if value_to_use is not None:
            exp_name = value_to_use.name
            experiment_dict[exp_name] = value_to_use
    return experiment_dict
