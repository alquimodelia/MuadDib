import glob
import inspect
import os

import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR

from muaddib.models.default_function import (
    keras_train_model,
    statsmodel_train_model,
)
from muaddib.muaddib import ShaiHulud
from muaddib.shaihulud_utils import (
    check_trained_epochs,
    expand_all_alternatives,
    list_folders,
    load_json_dict,
    open_model,
    write_dict_to_file,
)


def default_muaddib_model_builder(params, filepath=None):
    from forecat.models_definitions import get_model_from_def

    model_arch = params.pop("archs")
    archs_args = {}
    archs_args["filters"] = params.pop("filters")
    model_obj = get_model_from_def(
        model_arch, input_args=params, architecture_args=archs_args
    )
    if filepath:
        model_json = model_obj.to_json()
        write_dict_to_file(model_json, filepath)


class BaseModelHandler(ShaiHulud):
    register = set()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Check if the class has the required methods and attribute
        if all(
            hasattr(cls, method)
            for method in ["train_model", "validate_model"]
        ) and (hasattr(cls, "models_confs")):
            # If the criteria are met, add the class to the registry
            BaseModelHandler.register.add(cls)

    def __init__(
        self,
        name=None,
        target_variable=None,
        train_fn=None,
        datamanager=None,
        **kwargs,
    ):
        self.name = name
        self.work_folder = os.path.join(
            self.project_manager.trained_models_folder,
            datamanager.target_variable,
            datamanager.name,
        )

        self.target_variable = target_variable

        self.train_fn = train_fn
        self.datamanager = datamanager

        super().__init__(work_folder=self.work_folder, **kwargs)

    def read_model_file(self):
        raise NotImplementedError

    def save_model_file(self):
        raise NotImplementedError

    def list_models_confs(self):
        raise NotImplementedError

    def train_model(self):
        raise NotImplementedError

    def validate_model(self):
        raise NotImplementedError


class KerasModelHandler(BaseModelHandler):
    models_confs = None
    model_args = [
        "activation_end",
        "activation_middle",
        "X_timeseries",
        "Y_timeseries",
        "filters",
        "n_features_train",
        "n_features_predict",
    ]
    model_archs = ["CNN", "LSTM", "UNET", "Transformer", "Dense"]

    def assert_arch_in_class(arch):
        for ar in [f.lower() for f in KerasModelHandler.model_archs]:
            if ar in arch.lower():
                return True
        return False

    def __init__(
        self,
        archs=None,
        activation_middle=None,
        activation_end=None,
        X_timeseries=None,
        Y_timeseries=None,
        filters=None,
        n_features_train=None,
        n_features_predict=None,
        data_manager_name="",
        keras=True,
        project_manager=None,
        **kwargs,
    ):
        """

        Arch/activation_middle/activation_end

        Constructor method.

        Parameters
        ----------
        p1 : str, optional
            Description of the parameter, by default "whatever"
        """
        self.project_manager = project_manager
        self.configuration_folder = project_manager.model_configuration_folder

        self.archs = archs
        self.activation_middle = activation_middle
        self.activation_end = activation_end

        self.X_timeseries = X_timeseries
        self.Y_timeseries = Y_timeseries
        self.filters = filters

        self.n_features_train = n_features_train
        self.n_features_predict = n_features_predict

        self.keras = keras

        self.models_confs_list = self.list_models_confs()
        self.models_confs = self.name_models()
        self.models_to_train = []

        super().__init__(**kwargs)

    def list_models_confs(self):
        archs = self.archs or "VanillaCNN"
        activation_middle = self.activation_middle or "relu"
        activation_end = self.activation_end or "relu"

        X_timeseries = self.X_timeseries or 168
        Y_timeseries = self.Y_timeseries or 24
        filters = self.filters or 16

        n_features_predict = self.n_features_predict or 1
        n_features_train = self.n_features_train or 18

        parameters_to_list = {
            "archs": archs,
            "activation_middle": activation_middle,
            "activation_end": activation_end,
            "X_timeseries": X_timeseries,
            "Y_timeseries": Y_timeseries,
            "filters": filters,
            "n_features_predict": n_features_predict,
            "n_features_train": n_features_train,
        }
        return expand_all_alternatives(parameters_to_list)

    def name_models(self, models_names=None):
        named_models = {}
        for i, mod in enumerate(self.models_confs_list):
            if models_names:
                mod_name = models_names[i]
            else:
                ar = mod["archs"]
                am = mod["activation_middle"]
                ae = mod["activation_end"]
                x = mod["X_timeseries"]
                y = mod["Y_timeseries"]
                f = mod["filters"]
                T = mod["n_features_train"]
                P = mod["n_features_predict"]

                mod_name = f"{ar}_{am}_{ae}_X{x}_Y{y}_f{f}_T{T}_P{P}"
            named_models[mod_name] = mod

            # Create conf file for model
            file_conf_path = os.path.join(
                self.configuration_folder, f"{mod_name}.json"
            )
            if not os.path.exists(file_conf_path):
                default_muaddib_model_builder(mod, filepath=file_conf_path)
        return named_models

    def get_model_obj(self, model_case_name, loss=None):
        trained_models = list_folders(self.work_folder)
        model_conf_name = [
            f for f in self.models_confs.keys() if f in model_case_name
        ][0]
        model_obj_path = os.path.join(
            self.configuration_folder, f"{model_conf_name}.json"
        )
        last_epoch = 0
        trained_folder = os.path.join(self.work_folder, model_case_name)
        freq_saves_folder = os.path.join(trained_folder, "freq_saves")
        os.makedirs(freq_saves_folder, exist_ok=True)

        if model_case_name in trained_models:
            os.makedirs(freq_saves_folder, exist_ok=True)
            last_epoch, last_epoch_path = check_trained_epochs(
                freq_saves_folder
            )
            model_obj_path = last_epoch_path or model_obj_path
        custom_objects = {}
        if loss:
            custom_objects.update({"loss": loss})
        model_obj = open_model(
            model_obj_path, custom_objects=custom_objects, compile_arg=False
        )

        return model_obj, last_epoch, freq_saves_folder

    def set_callbacks(self, callbacks, freq_saves_path=None, last_epoch=0):
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        loaded_callbacks = []
        for callback in callbacks:
            callback_args = {}
            arg_names = inspect.getfullargspec(callback).args
            if "save_frequency" in arg_names:
                callback_args["save_frequency"] = 1
            if "start_epoch" in arg_names:
                callback_args["start_epoch"] = last_epoch
            if "model_keras_filename" in arg_names:
                frq_model_filename_sof = (
                    f"{freq_saves_path}" + "/{epoch}.keras"
                )
                callback_args["model_keras_filename"] = frq_model_filename_sof
            if "filepath" in arg_names:
                callback_args["filepath"] = os.path.dirname(freq_saves_path)
            if "model_log_filename" in arg_names:
                callback_args["model_log_filename"] = os.path.join(
                    os.path.dirname(freq_saves_path), "model_log.json"
                )
            if "logs" in arg_names:
                if os.path.exists(
                    os.path.join(
                        os.path.dirname(freq_saves_path), "model_log.json"
                    )
                ):
                    history = load_json_dict(
                        os.path.join(
                            os.path.dirname(freq_saves_path), "model_log.json"
                        )
                    )
                    callback_args["logs"] = history
            loaded_callbacks.append(callback(**callback_args))
        return loaded_callbacks

    def train_model(
        self,
        model_case_name,
        epochs,
        datamanager=None,
        train_fn=None,
        callbacks=None,
        **kwargs,
    ):
        loss = kwargs.get("loss", None)
        model_obj, last_epoch, freq_saves_folder = self.get_model_obj(
            model_case_name, loss=loss
        )
        if last_epoch:
            epochs = epochs - last_epoch
        if epochs < 1:
            return
        if callbacks:
            callbacks = self.set_callbacks(
                callbacks, freq_saves_folder, last_epoch
            )
            kwargs["callbacks"] = callbacks
        train_fn = train_fn or self.train_fn or keras_train_model
        datamanager = datamanager or self.datamanager

        train_fn(model_obj, epochs, datamanager, **kwargs)

    def validate_model(
        self,
        model_case_name,
        validation_fn,
        datamanager,
        old_score=None,
        model_types=".keras",
        **kwargs,
    ):
        trained_folder = os.path.join(
            self.work_folder, model_case_name, "freq_saves"
        )
        if old_score is not None:
            model_scores = old_score
        else:
            model_scores = pd.DataFrame()
        if "epoch" not in model_scores:
            model_scores["epoch"] = None
        for trained_models in glob.glob(f"{trained_folder}/**{model_types}"):
            epoca = int(
                os.path.basename(trained_models).replace(f"{model_types}", "")
            )
            if epoca not in model_scores["epoch"].values:
                predict_score = validation_fn(
                    trained_models,
                    datamanager,
                    model_name=model_case_name,
                    **kwargs,
                )
                predict_score["epoch"] = epoca
                model_scores = pd.concat(
                    [model_scores, pd.DataFrame(predict_score)]
                )
        return model_scores.reset_index(drop=True)


class StatsModelHandler(BaseModelHandler):
    """p P or lags: This is the order of the AutoRegressive (AR) part of the model. It indicates the number of lag observations that should be used as predictors. In other words, it specifies how many previous time points should be used to predict the current time point.
        d D: This is the order of differencing required to make the time series stationary. Differencing is a method used to remove trends and seasonality from time series data. The value of D represents the number of times the data has been differenced. # if it is stationary then 0
        q Q: This is the order of the Moving Average (MA) part of the model. It indicates the number of lagged forecast errors that should be used as predictors. The MA part of the model helps to capture the error terms of the AR part.
        s: This is the seasonal period of the time series. It is used in the seasonal part of the ARIMA model (SARIMA) to capture seasonal patterns. The value of S indicates the number of periods in each season.
        Seasonal orders (P, D, Q, S): These are the orders for the seasonal part of the ARIMA model. They are used when the time series exhibits seasonal patterns. The seasonal orders are similar to the non-seasonal orders but are applied to the seasonal component of the time series.

        trend: {'n', 'c', 't', 'ct'} The trend to include in the model:
            'n' - No trend.
            'c' - Constant only.
            't' - Time trend only.
            'ct' - Constant and time trend.


    AR (AutoRegressive):
    Order: (var, 0, 0)
    Seasonal Order: (0, 0, 0, 0)

    MA (Moving Average):
    Order: (0, 0, var)
    Seasonal Order: (0, 0, 0, 0)

    ARMA (AutoRegressive Moving Average):
    Order: (var, 0, var)
    Seasonal Order: (0, 0, 0, 0)

    ARIMA (AutoRegressive Integrated Moving Average):
    Order: (var, var, var)
    Seasonal Order: (0, 0, 0, 0)

    SARIMA (Seasonal AutoRegressive Integrated Moving Average):
    Order: (var, var, var)
    Seasonal Order: (var, var, var, var)


    """

    models_confs = None
    model_args = [
        "p",
        "d",
        "q",
        "P",
        "D",
        "Q",
        "s",
        "trend",
    ]
    model_archs = ["AR", "MA", "ARMA", "ARIMA", "SARIMA"]

    def assert_arch_in_class(arch):
        if arch.lower() in [f.lower() for f in StatsModelHandler.model_archs]:
            return True
        return False

    def __init__(
        self,
        p=None,
        d=None,
        q=None,
        P=None,
        D=None,
        Q=None,
        s=None,
        trend=None,
        project_manager=None,
        archs=None,
        **kwargs,
    ):
        self.archs = archs
        self.trend = trend
        self.p = p
        self.d = d
        self.q = q
        self.P = P
        self.D = D
        self.Q = Q
        self.s = s

        self.project_manager = project_manager
        self.configuration_folder = project_manager.model_configuration_folder
        self.models_confs_list = self.list_models_confs()
        self.models_confs = self.name_models()

        super().__init__(**kwargs)

    def get_stats_model(self, arch):
        arch = arch.lower()
        stats_model = None
        model_args = {}
        trend = self.trend
        if trend:
            model_args["trend"] = trend
        model_args["model_name"] = arch

        if arch == "ar":
            lags = self.p
            model_args["lags"] = lags
            model_args["mode_fn"] = AutoReg
            expanded_models_list = expand_all_alternatives(model_args)
        if arch == "ma":
            model_args["mode_fn"] = ARIMA
            model_args["q"] = self.q
            expanded_models_list = expand_all_alternatives(model_args)
            new_expanded_models_list = []
            for case_model in expanded_models_list:
                q = case_model.pop("q", 1) or 1
                case_model["order"] = (0, 0, q)
                new_expanded_models_list.append(case_model)
            expanded_models_list = new_expanded_models_list
        if arch == "arma":
            model_args["mode_fn"] = ARIMA
            model_args["q"] = self.q
            model_args["p"] = self.p
            expanded_models_list = expand_all_alternatives(model_args)
            new_expanded_models_list = []
            for case_model in expanded_models_list:
                q = case_model.pop("q", 1) or 1
                p = case_model.pop("p", 1) or 1
                case_model["order"] = (p, 0, q)
                new_expanded_models_list.append(case_model)
            expanded_models_list = new_expanded_models_list
        if arch == "arima":
            model_args["mode_fn"] = ARIMA
            model_args["q"] = self.q
            model_args["p"] = self.p
            model_args["d"] = self.d

            expanded_models_list = expand_all_alternatives(model_args)
            new_expanded_models_list = []
            for case_model in expanded_models_list:
                q = case_model.pop("q", 1) or 1
                p = case_model.pop("p", 1) or 1
                d = case_model.pop("d", 1) or 1
                case_model["order"] = (p, d, q)
                new_expanded_models_list.append(case_model)
            expanded_models_list = new_expanded_models_list

        if arch == "sarima":
            model_args["mode_fn"] = ARIMA
            model_args["q"] = self.q
            model_args["p"] = self.p
            model_args["d"] = self.d
            model_args["Q"] = self.Q
            model_args["P"] = self.P
            model_args["D"] = self.D

            expanded_models_list = expand_all_alternatives(model_args)
            new_expanded_models_list = []
            for case_model in expanded_models_list:
                q = case_model.pop("q", 1) or 1
                p = case_model.pop("p", 1) or 1
                d = case_model.pop("d", 1) or 1
                case_model["order"] = (p, d, q)
                Q = case_model.pop("Q", 1) or 1
                P = case_model.pop("P", 1) or 1
                D = case_model.pop("D", 1) or 1
                s = case_model.pop("s", 1) or 1

                case_model["seasonal_order"] = (P, D, Q, s)

                new_expanded_models_list.append(case_model)
            expanded_models_list = new_expanded_models_list

        if arch == "var model":
            model_args["mode_fn"] = VAR

        return expanded_models_list

    def list_models_confs(self):
        print(self.__dict__)
        archs = self.archs or "AR"
        if not isinstance(archs, list):
            archs = [archs]

        expanded_alternatives = []
        for arch in archs:
            arch_expanded = self.get_stats_model(arch)
            for a_exp in arch_expanded:
                expanded_alternatives.append(a_exp)

        return expanded_alternatives

    def name_models(self):
        self.models_confs = {}
        for model_args in self.models_confs_list:
            model_name = model_args.get("model_name")
            p, d, q = model_args.get("order", (0, 0, 0))
            if model_args.get("lags", None):
                p = model_args["lags"]
            P, D, Q, s = model_args.get("seasonal_order", (0, 0, 0, 0))

            case_name = f"{model_name}_p{p}_d{d}_q{q}_P{P}_D{D}_Q{Q}_s{s}"
            self.models_confs[case_name] = model_args

    def train_model(
        self,
        model_case_name,
        epochs,
        datamanager=None,
        train_fn=None,
        model_fn=None,
        order=None,
        seasonal_order=None,
        lags=None,
        trend=None,
        **kwargs,
    ):
        train_fn = train_fn or self.train_fn or statsmodel_train_model
        model_args = {}
        if order:
            model_args["order"] = order
        if seasonal_order:
            model_args["seasonal_order"] = seasonal_order
        if lags:
            model_args["lags"] = lags
        if trend:
            model_args["trend"] = trend
        # TODO: check UNIVARIATE??
        model_obj = model_fn(
            datamanager.read_data()[self.target_variable], **model_args
        )
        modelfilepath = os.path.join(
            self.work_folder,
            self.target_variable,
            model_case_name,
            "modelfit.pkl",
        )
        if not os.path.exists(modelfilepath):
            train_fn(model_obj=model_obj, modelfilepath=modelfilepath)

        return

    def validate_model(
        self,
    ):
        return


class ModelHandler(ShaiHulud):
    registry = dict()

    def __init__(self):
        pass

    def __new__(cls, model_backend: str, **kwargs):
        # Dynamically create an instance of the specified model class
        model_backend = model_backend.lower()
        modelhandler_class = ModelHandler.registry[model_backend]
        instance = super().__new__(modelhandler_class)
        # Inspect the __init__ method of the model class to get its parameters
        init_params = inspect.signature(cls.__init__).parameters
        # Separate kwargs based on the parameters expected by the model's __init__
        modelhandler_kwargs = {
            k: v for k, v in kwargs.items() if k in init_params
        }
        modelbackend_kwargs = {
            k: v for k, v in kwargs.items() if k not in init_params
        }

        for name, method in cls.__dict__.items():
            if "__" in name:
                continue
            if callable(method) and hasattr(instance, name):
                instance.__dict__[name] = method.__get__(instance, cls)

        cls.__init__(instance, **modelhandler_kwargs)
        instance.__init__(**modelbackend_kwargs)

        return instance

    @staticmethod
    def register(constructor):
        # TODO: only register if its a BaseModel subclass
        ModelHandler.registry[
            constructor.__name__.lower().replace("handler", "")
        ] = constructor

    @classmethod
    def create_model_handlers(cls, **kwargs):
        model_archs_hander = dict()
        archs = kwargs.pop("archs", [])
        if not isinstance(archs, list):
            archs = [archs]
        for (
            model_handler_name,
            model_handler_cls,
        ) in ModelHandler.registry.items():
            for arch in archs:
                if model_handler_cls.assert_arch_in_class(arch):
                    if model_handler_name not in model_archs_hander:
                        model_archs_hander[model_handler_name] = [arch]
                    else:
                        model_archs_hander[model_handler_name].append(arch)

        model_handlers_to_return = dict()
        model_handlers_kwargs = dict()

        kwargs_original = {**kwargs}

        for model_handler_name, archs in model_archs_hander.items():
            kwargs_copy = {**kwargs}
            model_handler_kwargs = {
                key: kwargs_copy[key]
                for key in ModelHandler.registry[model_handler_name].model_args
                if key in kwargs_copy
            }
            for key in model_handler_kwargs.keys():
                kwargs_original.pop(key)
            model_handler_kwargs["archs"] = archs
            model_handlers_kwargs[model_handler_name] = model_handler_kwargs

        for (
            model_handler_name,
            model_handler_kwargs,
        ) in model_handlers_kwargs.items():
            model_handler_kwargs.update(kwargs_original)

            model_handlers_to_return[
                model_handler_name
            ] = ModelHandler.registry[model_handler_name](
                **model_handler_kwargs
            )
        return model_handlers_to_return


ModelHandler.register(KerasModelHandler)
ModelHandler.register(StatsModelHandler)
