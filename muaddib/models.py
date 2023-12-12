import glob
import itertools
import os

from muaddib.shaihulud_utils import (
    is_jsonable,
    load_json_dict,
    open_model,
    write_dict_to_file,
)

MODELS_FOLDER = os.getenv("MODELS_FOLDER", None)

X_TIMESERIES = os.getenv("X_TIMESERIES", 168)
Y_TIMESERIES = os.getenv("Y_TIMESERIES", 24)

TARGET_VARIABLE = os.getenv("TARGET_VARIABLE")

MODELS_ARCHS = {
    "StackedCNN": {
        "arch": "CNNArch",
        "architecture_args": {"block_repetition": 2},
    },
    "StackedDense": {
        "arch": "DenseArch",
        "architecture_args": {"block_repetition": 2},
    },
    "VanillaCNN": {
        "arch": "CNNArch",
    },
    "VanillaDense": {"arch": "DenseArch"},
    "VanillaLSTM": {"arch": "LSTMArch"},
    "StackedLSTMA": {
        "arch": "LSTMArch",
        "architecture_args": {"block_repetition": 2},
    },
    "UNET": {"arch": "UNETArch"},
    "EncoderDecoder": {"arch": "EncoderDecoder"},
}


class CaseModel:
    def __init__(
        self,
        MODEL_CONF_FOLDER=MODELS_FOLDER,
        name=None,
        arquitecture=None,
        # Data speciifics
        X_timeseries=X_TIMESERIES,
        Y_timeseries=Y_TIMESERIES,
        n_features_predict=None,
        n_features_train=None,
        activation_middle="relu",
        activation_end="relu",
        keras_backend="torch",
        conf_file=None,
        case_model_folder=None,
        model_keras_file=None,
        n_filters=16,
        model_types=".keras",
        case_to_study_name=None,
    ):
        self.MODEL_CONF_FOLDER = MODEL_CONF_FOLDER

        self.arquitecture = arquitecture

        self.model_types = model_types

        # Data speciifics
        self.X_timeseries = X_timeseries
        self.Y_timeseries = Y_timeseries
        self.activation_middle = activation_middle
        self.activation_end = activation_end
        self.n_features_predict = n_features_predict
        self.n_features_train = n_features_train
        self.n_filters = n_filters

        self.keras_backend = keras_backend

        self.conf_file = conf_file

        self.CASE_MODEL_FOLDER = case_model_folder

        self.model_keras_file = model_keras_file
        self.name = name or self.get_name()
        self.case_to_study_name = case_to_study_name

        self.input_args = {
            "X_timeseries": self.X_timeseries,
            "Y_timeseries": self.Y_timeseries,
            "n_features_train": self.n_features_train,
            "n_features_predict": self.n_features_predict,
            "activation_middle": self.activation_middle,
            "activation_end": self.activation_end,
        }

        self.set_model_conf()

    def get_name(self):
        name = self.arquitecture
        list_vars = [self.activation_middle, self.activation_end]
        for var in list_vars:
            var_to_add = str(var) or ""
            name += f"_{var_to_add}"
        x_timeseries = f"_X{self.X_timeseries}"
        y_timeseries = f"_Y{self.Y_timeseries}"
        n_features_predict = f"_P{self.n_features_predict}"
        n_features_train = f"_T{self.n_features_train}"
        n_filters = f"_f{self.n_filters}"
        name = (
            name
            + x_timeseries
            + y_timeseries
            + n_features_train
            + n_features_predict
            + n_filters
        )
        return name

    def set_model_conf(self, architecture_args=None, input_args=None):
        if self.conf_file is None:
            self.conf_file = os.path.join(
                self.MODEL_CONF_FOLDER, f"{self.name}.json"
            )
        if not os.path.exists(self.conf_file):
            architecture_args = architecture_args or {}
            if self.n_filters:
                architecture_args.update(
                    {"conv_args": {"filters": self.n_filters}}
                )
            input_args = input_args or self.input_args

            model_name = self.arquitecture

            import forecat

            # Create model
            model_conf = MODELS_ARCHS[model_name]
            model_conf_name = model_conf["arch"]
            model_conf_arch = getattr(forecat, model_conf_name)

            architecture_args_to_use = architecture_args.copy()
            architecture_args_to_use.update(
                model_conf.get("architecture_args", {})
            )
            architecture_args_to_use.update({"name": self.name})
            # TODO: change the filter things in forecat now it wont be possible to experiment on different filters
            architecture_args_to_use.pop("conv_args")

            forearch = model_conf_arch(**input_args)
            model_obj = forearch.architecture(**architecture_args_to_use)
            model_json = model_obj.to_json()
            write_dict_to_file(model_json, self.conf_file)

    def check_trained_epochs(self):
        # Checks how many epochs were trained
        list_query = f"{self.freq_saves_path}/**{self.model_types}"
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
            last_epoch_path = os.path.join(
                f"{self.freq_saves_path}", f"{last_epoch}{self.model_types}"
            )
        self.last_epoch_path = last_epoch_path
        self.last_epoch = last_epoch

    def set_case_model(self, loss=None, CASE_MODEL_FOLDER=None):
        self.CASE_MODEL_FOLDER = self.CASE_MODEL_FOLDER or CASE_MODEL_FOLDER
        self.case_work_folder = os.path.join(self.CASE_MODEL_FOLDER, self.name)
        self.freq_saves_path = os.path.join(
            self.CASE_MODEL_FOLDER, "freq_saves"
        )
        os.makedirs(self.freq_saves_path, exist_ok=True)
        self.check_trained_epochs()
        model_obj = None
        custom_objects = {"loss": loss}
        if self.last_epoch_path:
            model_obj = open_model(
                self.last_epoch_path, custom_objects=custom_objects
            )
        else:
            model_obj = open_model(self.conf_file)
        self.model_obj = model_obj


class ModelHalleck:
    def __init__(
        self,
        MODEL_CONF_FOLDER=MODELS_FOLDER,
        name=None,
        archs_to_use=None,
        # Data speciifics
        X_timeseries=None,
        Y_timeseries=None,
        n_features_predict=None,
        n_features_train=None,
        activation_middle=None,
        activation_end=None,
        keras_backend="torch",
        experiment_model_folder=None,
        n_filters=None,
        model_types=".keras",
        previous_halleck=None,
        best_model=None,
        conf_file=None,
    ):
        self.MODEL_CONF_FOLDER = MODEL_CONF_FOLDER

        self.archs_to_use = archs_to_use
        self.model_types = model_types

        # Data speciifics
        self.X_timeseries = X_timeseries
        self.Y_timeseries = Y_timeseries
        self.activation_middle = activation_middle
        self.activation_end = activation_end
        self.n_features_predict = n_features_predict
        self.n_features_train = n_features_train
        self.n_filters = n_filters

        self.keras_backend = keras_backend

        self.EXPERIMENT_MODEL_FOLDER = experiment_model_folder
        self.conf_file = conf_file
        self.previous_halleck = previous_halleck
        # self.name = name
        # self.work_folder = work_folder
        # self.raw_data_folder = raw_data_folder
        # self.processed_data_folder = processed_data_folder
        # self.dataset_file_name = dataset_file_name

        # # Data speciifics
        # self.X_timeseries = X_timeseries
        # self.Y_timeseries = Y_timeseries
        # self.activation_middle = activation_middle
        # self.activation_end = activation_end

        # self.datetime_col = datetime_col
        # self.columns_Y = columns_Y

        # self.keras_backend = keras_backend
        # self.process_complete = False

        # self.process_fn = process_fn
        # self.read_fn = read_fn

        # self.keras_sequence_cls = keras_sequence_cls
        # self.models_dict=models_dict
        # self.setup()

    def save(self, path=None):
        dict_to_load = self.__dict__.copy()
        dict_to_save = {}
        for key, value in dict_to_load.items():
            dict_to_save[key] = value
            if value is None:
                continue

            if key == "models_to_experiment":
                dict_to_save[key] = [v.name for k, v in value.items()]

            if not is_jsonable(dict_to_save[key]):
                dict_to_save[key] = (
                    dict_to_save[key].name
                    if hasattr(dict_to_save[key], "name")
                    else str(dict_to_save[key])
                )
        write_dict_to_file(dict_to_save=dict_to_save, path=path)

    def load(self, path=None):
        path = path or self.conf_file
        dict_to_load = load_json_dict(path)
        # with open(path, "r") as f:
        #     dict_to_load = json.load(f)
        # Create a new dictionary to store the loaded data
        dict_to_restore = {}

        # Restore the original data structures and objects
        for key, value in dict_to_load.items():
            dict_to_restore[key] = value
            if value is None:
                continue
            elif key == "models_to_experiment":
                # Load a Case
                dict_to_restore[key] = {k: CaseModel(name=k) for k in value}
        # if "previous_cases" in dict_to_restore:
        #     if dict_to_restore["previous_cases"]:
        #         previous_cases = []
        #         experiments_in_list = []
        #         for f in dict_to_restore["previous_cases"]:
        #             target_name, exp_name, case_name = f.split(":")
        #             experiments_in_list.append(f"{target_name}:{exp_name}")

        #         experiments_in_list = np.unique(experiments_in_list)
        #         cases_per_experiment = {}
        #         for exp in experiments_in_list:
        #             cases_per_experiment[exp] = []
        #             for f in dict_to_restore["previous_cases"]:
        #                 target_name, exp_name, case_name = f.split(":")
        #                 if f"{target_name}:{exp_name}" == exp:
        #                     cases_per_experiment[exp].append(case_name)
        #         previous_cases += [
        #                     case_obj
        #                     for case_name, case_obj in exp_obj.study_cases.items()
        #                     if case_name in cases_per_experiment[exp]
        #                 ]
        #         dict_to_restore["previous_cases"] = previous_cases

        self.__dict__.update(dict_to_restore)

    def creat_models_on_experiment(self):
        # TODO: keep track of what is being studied
        experiment_variables = set()
        what_is_on_study = set()

        # create the model case for each model
        # archs to use
        list_vars = [
            "activation_middle",
            "activation_end",
            "X_timeseries",
            "Y_timeseries",
            "n_filters",
            "n_features_train",
            "n_features_predict",
        ]
        self.input_args = {
            "X_timeseries": self.X_timeseries,
            "Y_timeseries": self.Y_timeseries,
            "n_features_train": self.n_features_train,
            "n_features_predict": self.n_features_predict,
            "activation_middle": self.activation_middle,
            "activation_end": self.activation_end,
        }
        previous_archs = None
        if self.previous_halleck:
            previous_archs = self.previous_halleck.best_archs
        archs_to_use = self.archs_to_use or previous_archs
        if not isinstance(archs_to_use, list):
            archs_to_use = [archs_to_use]
        else:
            what_is_on_study.add("archs")
        # hierarqy
        # 1 -list of vars
        # 2 - previous
        # 3 - input
        # Check if there is a previous, if so
        refactor_combinations = {}
        for var in list_vars:
            self_var = getattr(self, var)
            # If its none we want the next step to use default values
            if self_var is None:
                continue
            if isinstance(self_var, list):
                refactor_combinations[var] = self_var
                what_is_on_study.add(var)
                experiment_variables.add(self_var)

            else:
                var_to_use = None
                if self.previous_halleck:
                    var_to_use = getattr(self.previous_halleck, var)
                var_to_use = var_to_use or self_var
                refactor_combinations[var] = [var_to_use]

        # Generate all combinations of values
        combinations = list(itertools.product(*refactor_combinations.values()))

        # Create a new dictionary for each combination
        result_combinations = [
            dict(zip(refactor_combinations.keys(), combination))
            for combination in combinations
        ]
        models_to_experiment = {}
        case_to_study_name = ""
        for arch in archs_to_use:
            if "archs" in what_is_on_study:
                case_to_study_name = arch
            for model_args in result_combinations:
                for k, n in model_args.items():
                    if isinstance(n, int):
                        name_to_add = str(n)
                    elif isinstance(n, str):
                        name_to_add = n
                    if k in what_is_on_study:
                        if len(case_to_study_name) == 0:
                            case_to_study_name = name_to_add
                        else:
                            case_to_study_name += f"_{name_to_add}"
                casemodelobj = CaseModel(
                    arquitecture=arch,
                    **model_args,
                    case_to_study_name=case_to_study_name,
                )
                models_to_experiment[casemodelobj.name] = casemodelobj
        self.models_to_experiment = models_to_experiment
        self.save(self.conf_file)

    def setup(self, conf_file=None):
        self.conf_file = self.conf_file or conf_file
        # if there is a conf file read if

        if os.path.exists(self.conf_file):
            self.load(path=self.conf_file)
        else:
            self.creat_models_on_experiment()

    def set_best_case_model(self, best_models):
        if not isinstance(best_models, list):
            best_models = [best_models]
        self.best_archs = best_models
        self.save(self.conf_file)

    # @staticmethod
    # def get_models_dict(
    #     models_strutres=models_strutres,
    #     architecture_args={},
    #     input_args={},
    # ):
    #     models_dict = {}
    #     input_args=input_args or self.input_args

    #     for model_name in models_strutres:
    #         model_def_path = os.path.join(MODELS_FOLDER, f"{model_name}.json")
    #         if not os.path.exists(model_def_path):
    #             # Model
    #             model_conf = models_strutres[model_name]
    #             model_conf_name = model_conf["arch"]
    #             model_conf_arch = getattr(forecat, model_conf_name)

    #             architecture_args_to_use = architecture_args.copy()
    #             architecture_args_to_use.update(
    #                 model_conf.get("architecture_args", {})
    #             )
    #             architecture_args_to_use.update({"name": model_name})
    #             forearch = model_conf_arch(**input_args)
    #             models_dict[model_name] = forearch.architecture(
    #                 **architecture_args_to_use
    #             )
    #             model_json = models_dict[model_name].to_json()
    #             with open(model_def_path, "w") as outfile:
    #                 outfile.write(model_json)
    #         else:
    #             with open(model_def_path, "r") as json_file:
    #                 loaded_model_json = json_file.read()

    #             models_dict[model_name] = keras_core.models.model_from_json(
    #                 loaded_model_json
    #             )

    # return models_dict

    # @staticmethod
    # def get_models_on_experiments(
    #     models_to_use,
    #     architecture_args={},
    #     input_args={},
    # ):

    #     input_args=input_args or self.input_args

    #     models_dict = {}
    #     models_to_use = [f.split("_")[0] for f in models_to_use]
    #     vars_lists = []
    #     for key, value in input_args.items():
    #         # Check if the value is a list
    #         if isinstance(value, list):
    #             # This give me a list of which keys are cases
    #             vars_lists.append(key)
    #         else:
    #             input_args[key] = [value]
    #         # Extract the lists from the dictionary
    #     lists = list(input_args.values())
    #     # Use itertools.product to get all combinations
    #     for combination in itertools.product(*lists):
    #         case_name = ""
    #         # Zip the keys with the combination
    #         args = {
    #             key: value for key, value in zip(input_args.keys(), combination)
    #         }
    #         X_timeseries = args["X_timeseries"]
    #         Y_timeseries = args["Y_timeseries"]
    #         X_in_vars = "X_timeseries" in vars_lists
    #         Y_in_vars = "Y_timeseries" in vars_lists
    #         time_in_vars = X_in_vars or Y_in_vars

    #         activation_middle = args["activation_middle"]
    #         activation_end = args["activation_end"]
    #         activation_middle_in_vars = "activation_middle" in vars_lists
    #         activation_end_in_vars = "activation_end" in vars_lists
    #         activation_in_vars = (
    #             activation_middle_in_vars or activation_end_in_vars
    #         )

    #         if args["X_timeseries"] < args["Y_timeseries"]:
    #             continue
    #         if time_in_vars:
    #             case_name += f"X{X_timeseries}_Y{Y_timeseries}_"
    #         if activation_in_vars:
    #             case_name += f"{activation_middle}_{activation_end}_"
    #         if case_name.endswith("_"):
    #             case_name = case_name[:-1]

    #         for model_to_use in models_to_use:
    #             model_to_fetch = model_to_use.split("_")[0]
    #             model_name = model_to_use + "_" + case_name

    #             model_def_path = os.path.join(MODELS_FOLDER, f"{model_name}.json")
    #             if not os.path.exists(model_def_path):
    #                 # Model
    #                 model_conf = models_strutres[model_to_fetch]
    #                 model_conf_name = model_conf["arch"]
    #                 model_conf_arch = getattr(forecat, model_conf_name)

    #                 architecture_args_to_use = architecture_args.copy()
    #                 architecture_args_to_use.update(
    #                     model_conf.get("architecture_args", {})
    #                 )
    #                 architecture_args_to_use.update({"name": model_name})
    #                 forearch = model_conf_arch(**args)
    #                 models_dict[model_name] = forearch.architecture(
    #                     **architecture_args_to_use
    #                 )
    #                 model_json = models_dict[model_name].to_json()
    #                 with open(model_def_path, "w") as outfile:
    #                     outfile.write(model_json)
    #             else:
    #                 with open(model_def_path, "r") as json_file:
    #                     loaded_model_json = json_file.read()

    #                 models_dict[model_name] = keras_core.models.model_from_json(
    #                     loaded_model_json
    #                 )
    #     return models_dict
