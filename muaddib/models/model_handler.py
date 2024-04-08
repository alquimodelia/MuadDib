import glob
import inspect
import os

import pandas as pd

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


class ModelHandler(ShaiHulud):
    def __init__(
        self,
        name=None,
        archs=None,
        activation_middle=None,
        activation_end=None,
        X_timeseries=None,
        Y_timeseries=None,
        filters=None,
        n_features_train=None,
        n_features_predict=None,
        target_variable=None,
        project_manager=None,
        data_manager_name="",
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
        self.name = name
        self.archs = archs
        self.activation_middle = activation_middle
        self.activation_end = activation_end

        self.X_timeseries = X_timeseries
        self.Y_timeseries = Y_timeseries
        self.filters = filters

        self.n_features_train = n_features_train
        self.n_features_predict = n_features_predict

        self.work_folder = os.path.join(
            project_manager.trained_models_folder,
            target_variable,
            data_manager_name,
        )
        self.configuration_folder = project_manager.model_configuration_folder

        self.target_variable = target_variable

        self.models_confs_list = self.list_models_confs()
        self.models_confs = self.name_models()
        self.models_to_train = []
        super().__init__(work_folder=self.work_folder, **kwargs)

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
        train_fn,
        datamanager,
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
