import glob
import inspect
import os

import pandas as pd
from alquimodelia.alquimodelia import ModelMagia
from keras.losses import MeanSquaredError
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR

from muaddib.models.default_function import (
    inference_model,
    keras_train_model,
    metric_scores_default,
    statsmodel_train_model,
    validate_model,
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

    model_arch = params.pop("archs")
    archs_args = {}
    archs_args["filters"] = params.pop("filters")
    model_obj = ModelMagia(model_arch, **params, **archs_args).model

    if filepath:
        model_json = model_obj.to_json()
        write_dict_to_file(model_json, filepath)


class BaseModelHandler(ShaiHulud):
    register = set()
    listing_conf_properties = ["exp_cases", "models_confs"]
    single_conf_properties = [
        "project_manager",
        "datamanager",
        "epochs",
        "callbacks",
        "metric_scores_fn",
        "train_fn",
        "validation_target",
        "validation_fn",
        "inference_fn",
    ]

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
        metric_scores_fn=None,
        validation_fn=None,
        inference_fn=None,
        **kwargs,
    ):
        self.name = name
        self.work_folder = os.path.join(
            self.project_manager.trained_models_folder,
            self.datamanager.target_variable,
            self.datamanager.name,
        )

        self.target_variable = self.datamanager.target_variable

        self.train_fn = train_fn
        self.validation_fn = validation_fn
        self.inference_fn = inference_fn
        self.metric_scores_fn = metric_scores_fn or metric_scores_default
        self.set_experiments()

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

    def set_experiments(self):
        raise NotImplementedError


class KerasModelHandler(BaseModelHandler):
    model_args = [
        "activation_end",
        "activation_middle",
        "x_timesteps",
        "y_timesteps",
        "filters",
        "num_features_to_train",
        "num_classes",
    ]
    fit_kwargs = [
        "optimizer",
        "loss",
        "batch_size",
        "weights",
    ]
    model_archs = ["CNN", "LSTM", "UNET", "Transformer", "Dense"]
    class_args = ["epochs", "archs", "callbacks", "metric_scores_fn"]

    name_builder_position_dict={
            "archs": 0,
            "activation_middle": 1,
            "activation_end": 2,
            "x_timesteps": 3,
            "y_timesteps": 4,
            "filters": 5,
            "features": 6,
            "classes": 7,
            "optimizer": 8,
            "loss": 9,
            "batch": 10,
            "weights": 11,
    }



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
        x_timesteps=None,
        y_timesteps=None,
        filters=None,
        num_features_to_train=None,
        num_classes=None,
        optimizer=None,
        loss=None,
        batch_size=None,
        weights=None,
        datamanager=None,
        keras=True,
        project_manager=None,
        epochs=None,
        callbacks=None,
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
        self.datamanager = datamanager

        self.configuration_folder = project_manager.model_configuration_folder

        self.archs = archs
        self.activation_middle = activation_middle
        self.activation_end = activation_end
        extra_y_timesteps = max([0, datamanager.commun_steps])
        y_timesteps = datamanager.y_timesteps + extra_y_timesteps

        self.x_timesteps = x_timesteps or datamanager.x_timesteps
        self.y_timesteps = y_timesteps or y_timesteps
        self.filters = filters

        self.num_features_to_train = (
            num_features_to_train or datamanager.num_features_to_train
        )
        self.num_classes = num_classes or datamanager.num_classes

        self.optimizer = optimizer
        self.loss = loss
        self.batch_size = batch_size
        self.weights = weights
        self.epochs = epochs

        self.callbacks = callbacks

        self.models_confs_list = self.list_models_confs()
        self.models_confs = self.name_models()
        self.models_to_train = []

        super().__init__(**kwargs)

    def list_models_confs(self):
        archs = self.archs or "VanillaCNN"
        activation_middle = self.activation_middle or "relu"
        activation_end = self.activation_end or "relu"

        x_timesteps = self.x_timesteps or 168
        y_timesteps = self.y_timesteps or 24
        filters = self.filters or 16

        num_classes = self.num_classes or 1
        num_features_to_train = self.num_features_to_train or 18

        parameters_to_list = {
            "archs": archs,
            "activation_middle": activation_middle,
            "activation_end": activation_end,
            "x_timesteps": x_timesteps,
            "y_timesteps": y_timesteps,
            "filters": filters,
            "num_classes": num_classes,
            "num_features_to_train": num_features_to_train,
        }
        return expand_all_alternatives(parameters_to_list)

    def set_experiments(self, models_names=None):
        optimizer = self.optimizer or "adam"
        loss = self.loss or MeanSquaredError()
        batch_size = self.batch_size or 128
        weights = self.weights or False

        parameters_to_list = {
            "optimizer": optimizer,
            "loss": loss,
            "batch_size": batch_size,
            "weights": weights,
        }
        experiment_lits = expand_all_alternatives(parameters_to_list)
        exp_cases = {}
        for i, mod in enumerate(experiment_lits):
            if models_names:
                mod_name = models_names[i]
            else:
                opt = mod["optimizer"]
                los = "".join([f[0] for f in mod["loss"].name.split("_")])
                bs = str(mod["batch_size"])
                wt = mod["weights"]
                mod_name = f"{opt}_{los}_B{bs}_{wt}"

            for model_name in self.models_confs.keys():
                case_name = f"{model_name}_{mod_name}"
                exp_cases[case_name] = mod
                exp_cases[case_name]["epochs"] = self.epochs
        self.exp_cases = exp_cases

    def name_models(self, models_names=None):
        named_models = {}
        for i, mod in enumerate(self.models_confs_list):
            if models_names:
                mod_name = models_names[i]
            else:
                ar = mod["archs"]
                am = mod["activation_middle"]
                ae = mod["activation_end"]
                x = mod["x_timesteps"]
                y = mod["y_timesteps"]
                f = mod["filters"]
                T = mod["num_features_to_train"]
                P = mod["num_classes"]

                mod_name = f"{ar}_{am}_{ae}_X{x}_Y{y}_f{f}_T{T}_P{P}"
            named_models[mod_name] = mod

            # Create conf file for model
            file_conf_path = os.path.join(
                self.configuration_folder, f"{mod_name}.json"
            )
            if not os.path.exists(file_conf_path):
                default_muaddib_model_builder(mod, filepath=file_conf_path)
        return named_models

    def conf_to_build_args(self, conf):
        build_args = dict()
        for arg in self.model_args + self.fit_kwargs + self.class_args:
            if arg in conf:
                build_args[arg] = conf[arg]

        return build_args

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
        epochs=None,
        datamanager=None,
        train_fn=None,
        callbacks=None,
        **kwargs,
    ):
        loss = kwargs.get("loss", None)
        model_obj, last_epoch, freq_saves_folder = self.get_model_obj(
            model_case_name, loss=loss
        )
        if isinstance(epochs, list):
            epochs=min(epochs)
        if last_epoch:
            # TODO: fix epochs from experiment to model handler, it is making a list of both
            epochs = epochs - last_epoch
        if epochs < 1:
            return
        callbacks = callbacks or self.callbacks
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
        datamanager,
        validation_fn=None,
        inference_fn=None,
        old_score=None,
        model_types=".keras",
        **kwargs,
    ):
        trained_folder = os.path.join(
            self.work_folder, model_case_name, "freq_saves"
        )
        validation_fn = validation_fn or self.validation_fn or validate_model
        inference_fn = inference_fn or self.inference_fn or inference_model
        prediction_score_path = trained_folder.replace(
            "freq_saves", "predictions_score.csv"
        )
        old_score_none = old_score is not None
        old_score_empty=False
        if old_score_none:
            old_score_empty = len(old_score)==0
        else:
            old_score_empty=True
        old_score_not_empty = not old_score_empty
        if old_score_not_empty:
            model_scores = old_score
        elif os.path.exists(prediction_score_path):
            model_scores = pd.read_csv(prediction_score_path)
        else:
            model_scores = pd.DataFrame()
        if "epoch" not in model_scores:
            model_scores["epoch"] = None
        for trained_models in glob.glob(f"{trained_folder}/**{model_types}"):
            epoca = int(
                os.path.basename(trained_models).replace(f"{model_types}", "")
            )

            prediction_path = os.path.dirname(trained_models).replace(
                "freq_saves", "predictions.npz"
            )
            prediction_name = f"prediction_{epoca}"
            if epoca not in model_scores["epoch"].values:
                prediction_file_dict = inference_fn(
                    prediction_path,
                    prediction_name,
                    trained_models,
                    datamanager=datamanager,
                    model_type="keras",
                    **kwargs,
                )
                # TODO: only do predict if there is none, maybe its hapening
                predict_score = validation_fn(
                    prediction_file_dict,
                    prediction_name,
                    metric_scores=self.metric_scores_fn,
                )

                predict_score["epoch"] = epoca
                predict_score["name"] = model_case_name
                predict_score.update(
                    self.return_dict_from_name(model_case_name)
                )
                model_scores = pd.concat(
                    [model_scores, pd.DataFrame(predict_score)]
                )
        model_scores = model_scores.reset_index(drop=True)
        model_scores.drop_duplicates(inplace=True)
        model_scores.to_csv(prediction_score_path, index=False)
        return model_scores

    def return_dict_from_name(self, name):
        (
            arch,
            activation_middle,
            activation_end,
            x_timesteps,
            y_timesteps,
            filters,
            features,
            classes,
            optimizer,
            loss,
            batch,
            weights,
        ) = name.split("_")
        return {
            "archs": arch,
            "activation_middle": activation_middle,
            "activation_end": activation_end,
            "x_timesteps": int(x_timesteps.replace("X", "")),
            "y_timesteps": int(y_timesteps.replace("Y", "")),
            "filters": int(filters.replace("f", "")),
            "features": int(features.replace("T", "")),
            "classes": int(classes.replace("P", "")),
            "optimizer": optimizer,
            "loss": loss,
            "batch": int(batch.replace("B", "")),
            "weights": weights,
        }


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
    fit_kwargs = []
    class_args = ["archs", "metric_scores_fn"]

    name_builder_position_dict={
            "archs": 0,
            "p": 1,
            "d": 2,
            "q": 3,
            "P": 4,
            "D": 5,
            "Q": 6,
            "s": 7,
            "trend": 8,
    }



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
        datamanager=None,
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

        self.datamanager = datamanager
        self.project_manager = project_manager
        self.configuration_folder = project_manager.model_configuration_folder
        self.models_confs_list = self.list_models_confs()
        self.name_models()

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
            model_args["model_fn"] = AutoReg
            expanded_models_list = expand_all_alternatives(model_args)
        if arch == "ma":
            model_args["model_fn"] = ARIMA
            model_args["q"] = self.q
            expanded_models_list = expand_all_alternatives(model_args)
            new_expanded_models_list = []
            for case_model in expanded_models_list:
                q = case_model.pop("q", 1) or 1
                case_model["order"] = (0, 0, q)
                new_expanded_models_list.append(case_model)
            expanded_models_list = new_expanded_models_list
        if arch == "arma":
            model_args["model_fn"] = ARIMA
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
            model_args["model_fn"] = ARIMA
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
            model_args["model_fn"] = ARIMA
            model_args["q"] = self.q
            model_args["p"] = self.p
            model_args["d"] = self.d
            model_args["Q"] = self.Q
            model_args["P"] = self.P
            model_args["D"] = self.D
            model_args["s"] = self.s

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
                s = case_model.pop("s", 24) or 24

                case_model["seasonal_order"] = (P, D, Q, s)

                new_expanded_models_list.append(case_model)
            expanded_models_list = new_expanded_models_list

        if arch == "var model":
            model_args["model_fn"] = VAR

        return expanded_models_list

    def list_models_confs(self):
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
            trend = model_args.get("trend", "n")

            case_name = (
                f"{model_name}_p{p}_d{d}_q{q}_P{P}_D{D}_Q{Q}_s{s}_t{trend}"
            )
            self.models_confs[case_name] = model_args

    def conf_to_build_args(self, conf):
        build_args = dict()
        build_args["archs"] = conf["model_name"]
        for arg in self.model_args + self.fit_kwargs + self.class_args:
            if arg in conf:
                build_args[arg] = conf[arg]
        for arg in conf.keys():
            if arg not in build_args:
                if arg == "lags":
                    build_args["p"] = conf[arg]
                if arg == "order":
                    p, d, q = conf[arg]
                    build_args["p"] = p
                    build_args["q"] = q
                    build_args["d"] = d
                if arg == "seasonal_order":
                    P, D, Q, s = conf[arg]
                    build_args["P"] = P
                    build_args["Q"] = Q
                    build_args["D"] = D
                    build_args["s"] = s

        return build_args

    def set_experiments(self, models_names=None):
        exp_cases = {}
        fit_args = {}
        for case_name, case_args in self.models_confs.items():
            exp_cases[case_name] = {**case_args}

        self.exp_cases = exp_cases

    def train_model(
        self,
        model_case_name,
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
        datamanager = datamanager or self.datamanager
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
            model_case_name,
            "modelfit.pkl",
        )
        os.makedirs(os.path.dirname(modelfilepath), exist_ok=True)
        if not os.path.exists(modelfilepath):
            train_fn(model_obj=model_obj, modelfilepath=modelfilepath)

        return

    def validate_model(
        self,
        model_case_name,
        datamanager,
        validation_fn=None,
        inference_fn=None,
        old_score=None,
        model_types=".pkl",
        **kwargs,
    ):
        trained_folder = os.path.join(self.work_folder, model_case_name)
        validation_fn = validation_fn or self.validation_fn or validate_model
        inference_fn = inference_fn or self.inference_fn or inference_model
        prediction_score_path = os.path.join(
            trained_folder, "predictions_score.csv"
        )
        if old_score is not None:
            model_scores = old_score
        elif os.path.exists(prediction_score_path):
            model_scores = pd.read_csv(prediction_score_path)
        else:
            model_scores = pd.DataFrame()
        if "epoch" not in model_scores:
            model_scores["epoch"] = None
        for trained_models in glob.glob(f"{trained_folder}/**{model_types}"):

            epoca = 1
            prediction_path = os.path.join(trained_folder, "predictions.npz")
            prediction_name = f"prediction_{epoca}"
            if epoca not in model_scores["epoch"].values:
                prediction_file_dict = inference_fn(
                    prediction_path,
                    prediction_name,
                    trained_models,
                    datamanager=datamanager,
                    model_type="statsmodel",
                    **kwargs,
                )
                predict_score = validation_fn(
                    prediction_file_dict,
                    prediction_name,
                    metric_scores=self.metric_scores_fn,
                )

                predict_score["epoch"] = 1
                predict_score["name"] = model_case_name
                predict_score.update(
                    self.return_dict_from_name(model_case_name)
                )
                model_scores = pd.concat(
                    [model_scores, pd.DataFrame(predict_score)]
                )
        model_scores = model_scores.reset_index(drop=True)
        model_scores.drop_duplicates(inplace=True)
        model_scores.to_csv(prediction_score_path, index=False)
        return model_scores

    def return_dict_from_name(self, name):
        arch, p, d, q, P, D, Q, s, trend = name.split("_")
        return {
            "archs": arch,
            "p": int(p.replace("p", "")),
            "d": int(d.replace("d", "")),
            "q": int(q.replace("q", "")),
            "P": int(P.replace("P", "")),
            "D": int(D.replace("D", "")),
            "Q": int(Q.replace("Q", "")),
            "s": int(s.replace("s", "")),
            "trend": trend,
        }


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
        name = kwargs.pop("name", "exp1")
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
            handler_name = f"{name}_{model_handler_name}_handler"
            model_handlers_to_return[model_handler_name] = ModelHandler(
                name=handler_name,
                model_backend=model_handler_name,
                **model_handler_kwargs,
            )
        return model_handlers_to_return

    @classmethod
    def return_child_by_arch(cls, arch):
        for (
            model_handler_name,
            model_handler_cls,
        ) in ModelHandler.registry.items():
            if model_handler_cls.assert_arch_in_class(arch):
                return model_handler_name



ModelHandler.register(KerasModelHandler)
ModelHandler.register(StatsModelHandler)
