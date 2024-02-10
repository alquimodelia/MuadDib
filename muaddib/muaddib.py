import copy
import glob
import inspect
import math
import os
import pathlib

import keras
import numpy as np
import pandas as pd
import tinydb
from keras.losses import MeanSquaredError

from muaddib.shaihulud_utils import (
    AdvanceLossHandler,
    check_trained_epochs,
    expand_all_alternatives,
    list_folders,
    load_json_dict,
    open_model,
    write_dict_to_file,
)
from muaddib.tinydb_serializers import create_class_serializer, serialization


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


class ShaiHulud:
    """
    A class that represents a model.

    Attributes
    ----------
    p1 : str
        Description of the attribute

    Methods
    -------
    my_method(p2)
        Description of the method
    """

    def __init__(
        self,
        work_folder=None,
        **kwargs,
    ):
        """
        Constructor method.

        Parameters
        ----------
        p1 : str, optional
            Description of the parameter, by default "whatever"
        """
        self.work_folder = work_folder or str(pathlib.Path("").resolve())
        conf_file = getattr(self, "conf_file", None)
        self.conf_file = conf_file
        if conf_file and os.path.exists(conf_file):
            self.load(conf_file)
        else:
            self.setup(**kwargs)

    def get_vars_to_save(self):
        return None

    def save(self):
        serialization.register_serializer(
            create_class_serializer(ShaiHulud), "ShaiHulud"
        )
        db = tinydb.TinyDB(
            self.conf_file, storage=serialization, indent=4, sort_keys=True
        )
        vars_to_save = self.get_vars_to_save() or vars(self)
        records = db.all()
        if records:
            record = records[0]  # assuming you only have one record
            record.update(vars_to_save)
            db.update(record)
        else:
            db.insert(vars_to_save)

    def setup_after_load(self):
        pass

    def load(self, conf_file):
        serialization.register_serializer(
            create_class_serializer(ShaiHulud), "ShaiHulud"
        )
        db = tinydb.TinyDB(
            conf_file, storage=serialization, indent=4, sort_keys=True
        )
        record = db.all()[0]  # assuming you only have one record
        for attr, value in record.items():
            setattr(self, attr, value)
        self.setup_after_load()

    def obj_setup(self, **kwargs):
        pass

    def setup(self, **kwargs):
        obj_setup_args = kwargs.pop("obj_setup_args", {})
        if not self.conf_file:
            filename = self.name or self.__class__.__name__
            self.conf_file = os.path.join(
                self.work_folder, f"{filename}_conf.json"
            )
            os.makedirs(os.path.dirname(self.conf_file), exist_ok=True)
        for kwarg, value in kwargs.items():
            setattr(self, str(kwarg), value)
        # Only setup the experiment if there is
        if not obj_setup_args.get("previous_experiment", False):
            self.obj_setup(**obj_setup_args)
        self.save()


class ProjectFolder(ShaiHulud):
    def __init__(
        self,
        name=None,
        root_folder=None,
        target_variables=None,
        **kwargs,
    ):
        self.name = name or "muaddib_project"
        root_folder = root_folder or "."
        self.root_folder = pathlib.Path(root_folder).absolute()
        self.target_variables = target_variables

        self.data_folder = self.root_folder.joinpath("data")
        self.experiment_folder = self.root_folder.joinpath("experiment")
        self.models_folder = self.root_folder.joinpath("models")
        self.notebooks_folder = self.root_folder.joinpath("notebooks")
        self.references_folder = self.root_folder.joinpath("references")
        self.reports_folder = self.root_folder.joinpath("reports")
        self.project_folder = self.root_folder.joinpath(self.name)

        self.model_configuration_folder = self.models_folder.joinpath(
            "configurations"
        )
        self.trained_models_folder = self.models_folder.joinpath(
            "trained_models"
        )

        self.trained_models_folder_variables = []
        self.reports_folder_variables = []
        self.experiments_variables_variables = []

        for target_variable in self.target_variables:
            self.trained_models_folder_variables.append(
                self.trained_models_folder.joinpath(target_variable)
            )
            self.reports_folder_variables.append(
                self.reports_folder.joinpath(target_variable)
            )
            self.experiments_variables_variables.append(
                self.experiment_folder.joinpath(target_variable)
            )

        super().__init__(work_folder=self.root_folder, **kwargs)
        self.check_and_mkdirs()

    def check_and_mkdirs(self):
        for name, var in vars(self).items():
            if isinstance(var, pathlib.Path):
                os.makedirs(var, exist_ok=True)
                setattr(self, name, str(var))
        for folder in (
            self.trained_models_folder_variables
            + self.reports_folder_variables
            + self.experiments_variables_variables
        ):
            if isinstance(folder, pathlib.Path):
                os.makedirs(folder, exist_ok=True)


class ExperimentHandler(ShaiHulud):
    def __init__(
        self,
        name=None,
        # Experiment
        optimizer=None,
        loss=None,
        batch_size=None,
        weights=None,
        # ModelHandler
        archs=None,
        activation_middle=None,
        activation_end=None,
        filters=None,
        target_variable=None,
        data_manager=None,
        project_manager=None,
        model_handler=None,
        train_fn=None,
        epochs=None,
        callbacks=None,
        validation_fn=None,
        result_validation_fn=None,
        validation_target=None,
        previous_experiment=None,
        **kwargs,
    ):
        """
        We want were to have batch/weights/loss/optimizer
        Constructor method.

        Parameters
        ----------
        p1 : str, optional
            Description of the parameter, by default "whatever"
        """
        conf_file = kwargs.get("conf_file", None)
        if conf_file and os.path.exists(conf_file):
            self.load(conf_file)
        else:
            self.optimizer = optimizer
            self.loss = loss
            self.batch_size = batch_size
            self.weights = weights

            self.target_variable = target_variable
            self.project_manager = project_manager

            self.name = name or "experiment1"

            self.work_folder = os.path.join(
                project_manager.experiment_folder,
                self.target_variable,
                self.name,
            )
            self.data_manager = data_manager
            self.train_fn = train_fn
            self.validation_fn = validation_fn
            self.result_validation_fn = result_validation_fn
            self.validation_target = validation_target

            self.epochs = epochs
            self.callbacks = callbacks

            model_handler_args = {
                "archs":archs,
                "activation_middle":activation_middle,
                "activation_end":activation_end,
                "X_timeseries":data_manager.X_timeseries,
                "Y_timeseries":data_manager.Y_timeseries,
                "filters":filters,

            }
            obj_setup_args = {
                "model_handler_args":model_handler_args,
                "previous_experiment":previous_experiment,
            }
            self.obj_setup_args=obj_setup_args
            # if previous_experiment:
            #     self.previous_case=previous_experiment.best_case
            #     previous_best_model = previous_experiment.model_handler.models_confs[previous_experiment.best_exp.model]
            #     previous_best_model.pop("n_features_predict")
            #     previous_best_model.pop("n_features_train")
            #     previous_best_model.update({k: v for k, v in model_handler_args.items() if v is not None})
            #     model_handler_args = previous_best_model

            # self.model_handler = ModelHandler(
            #     name=name,
            #     project_manager=project_manager,
            #     n_features_predict=data_manager.n_features_predict,
            #     n_features_train=data_manager.n_features_train,
            #     target_variable=self.target_variable,
            #     **model_handler_args,

            # )

            # self.experiment_list = self.list_experiments(previous_experiment)
            # self.experiments = self.get_experiment_models()

            super().__init__(
                obj_type="experiment", work_folder=self.work_folder, obj_setup_args=obj_setup_args,**kwargs
            )

    def obj_setup(self, model_handler_args=None, previous_experiment=None):
        model_handler_args = self.obj_setup_args.get("model_handler_args", model_handler_args)
        previous_experiment = self.obj_setup_args.get("previous_experiment", previous_experiment)
        if isinstance(self.loss, AdvanceLossHandler):
            self.loss.set_previous_loss(previous_experiment.best_exp["loss"])
        delattr(self, "obj_setup_args")


        if previous_experiment:
            self.previous_case=previous_experiment.best_case
            previous_best_model = previous_experiment.model_handler.models_confs[previous_experiment.best_exp["model"]]
            previous_best_model.pop("n_features_predict")
            previous_best_model.pop("n_features_train")
            previous_best_model.update({k: v for k, v in model_handler_args.items() if v is not None})
            model_handler_args = previous_best_model

        self.model_handler = ModelHandler(
            name=self.name,
            project_manager=self.project_manager,
            n_features_predict=self.data_manager.n_features_predict,
            n_features_train=self.data_manager.n_features_train,
            target_variable=self.target_variable,
            **model_handler_args,

        )

        self.experiment_list = self.list_experiments(previous_experiment)
        self.experiments = self.get_experiment_models()
        self.save()


    def list_experiments(self, previous_experiment=None):
        previous_experiment = previous_experiment or {}
        previous_best_exp = getattr(previous_experiment,"best_exp", {})

        optimizer = self.optimizer or previous_best_exp.get("optimizer", "adam")
        loss = self.loss or previous_best_exp.get("loss", MeanSquaredError())
        batch_size = self.batch_size or previous_best_exp.get("batch_size", 252)  
        weights = self.weights or previous_best_exp.get("weights", False)  

        parameters_to_list = {
            "optimizer": optimizer,
            "loss": loss,
            "batch_size": batch_size,
            "weights": weights,
        }
        return expand_all_alternatives(parameters_to_list)

    def name_experiments(self):
        named_exps = {}
        for exp in self.experiment_list:
            opt = exp["optimizer"]
            los = "".join([f[0] for f in exp["loss"].name.split("_")])
            bs = str(exp["batch_size"])
            wt = exp["weights"]
            exp_name = f"{opt}_{los}_B{bs}_{wt}"
            named_exps[exp_name] = exp
        return named_exps

    def get_experiment_models(self):
        experiments = {}
        for model_arch in self.model_handler.models_confs.keys():
            for case, case_args in self.name_experiments().items():
                case_name = f"{model_arch}_{case}"
                experiments[case_name] = case_args
                experiments[case_name]["model"] = model_arch
        return experiments

    def train_experiment(self):
        if not getattr(self, "experiments", False):
            self.obj_setup()
        for exp, exp_args in self.experiments.items():
            exp_fit_args = {**exp_args}
            if "model" in exp_fit_args:
                exp_fit_args.pop("model")
            self.model_handler.train_model(
                exp,
                self.epochs,
                self.train_fn,
                self.data_manager,
                callbacks=self.callbacks,
                **exp_fit_args,
            )

    def validate_experiment(self):
        exp_results_path = os.path.join(self.work_folder, "experiment_score.json")
        if os.path.exists(exp_results_path):
            exp_results = pd.read_json(load_json_dict(exp_results_path))
        else:
            exp_results = pd.DataFrame()
        # TODO: make a full exp socre file and only valdiate if needed
        for exp in self.experiments.keys():
            saved_exp_score = exp_results[exp_results.name==exp]
            if len(saved_exp_score)<self.epochs:
                exp_score = self.model_handler.validate_model(
                    exp, self.validation_fn, self.data_manager, old_score=saved_exp_score,
                )
                exp_results = pd.concat([exp_results, exp_score])
        exp_results = exp_results.drop_duplicates(["name","epoch"])    
        exp_results =  exp_results.reset_index(drop=True)
        write_dict_to_file(exp_results.to_json(), exp_results_path)
        return exp_results

    def validate_results(
        self,
        exp_results=None,
        result_validation_fn=None,
        validation_target=None,
        **kwargs
    ):
        if exp_results is None:
            exp_results = self.validate_experiment()
        result_validation_fn = (
            result_validation_fn or self.result_validation_fn
        )
        validation_target = validation_target or self.validation_target
        self.best_case, self.best_result = result_validation_fn(exp_results, validation_target, **kwargs)
        self.best_exp = self.experiments[self.best_case.name.item()]
        self.save()
        return self.best_case


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
        data_manager=None,
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
            project_manager.trained_models_folder, target_variable
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
        model_scores = old_score or pd.DataFrame()
        for trained_models in glob.glob(f"{trained_folder}/**{model_types}"):
            epoca = int(
                os.path.basename(trained_models).replace(f"{model_types}", "")
            )
            if epoca not  in model_scores["epoch"].values:
                predict_score = validation_fn(
                    trained_models,
                    datamanager,
                    model_name=model_case_name,
                )
                predict_score["epoch"] = epoca
                model_scores = pd.concat(
                    [model_scores, pd.DataFrame(predict_score)]
                )
        return model_scores.reset_index(drop=True)



# The data handle is quite good already
class DataHandler(ShaiHulud):
    def __init__(
        self,
        target_variable=None,
        dataset_file_name="dados_2014-2022.csv",  # TODO: change to something more general
        name=None,
        project_manager=None,
        # Data speciifics
        X_timeseries=168,
        Y_timeseries=24,
        columns_Y=None,  # List
        datetime_col="datetime",
        keras_backend="torch",
        process_fn=None,
        read_fn=None,
        validation_fn=None,
        process_benchmark_fn=None,
        keras_sequence_cls=None,
        sequence_args=None,
        score_path=None,
        **kwargs,
    ):
        """
        Initialize the DataHandler object.

        Args:
            target_variable: The target variable.
            dataset_file_name: The name of the dataset file. Defaults to "dados_2014-2022.csv".
            name: The name of the DataHandler object.
            project_manager: The project manager.

        Keyword Args:
            X_timeseries: The time series for X data. Defaults to 168.
            Y_timeseries: The time series for Y data. Defaults to 24.
            columns_Y: The list of columns for Y data.
            datetime_col: The name of the datetime column. Defaults to "datetime".
            keras_backend: The backend for Keras. Defaults to "torch".
            process_fn: The process function.
            read_fn: The read function.
            validation_fn: The validation function.
            process_benchmark_fn: The process benchmark function.
            keras_sequence_cls: The Keras sequence class.
            sequence_args: The sequence arguments.
            score_path: The path for the benchmark score.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        self.target_variable = target_variable
        columns_Y = columns_Y or [target_variable]
        self.project_manager = project_manager

        self.name = name or f"datahandle_{target_variable}"

        self.work_folder = project_manager.data_folder

        self.name = name
        self.raw_data_folder = os.path.join(self.work_folder, "raw")
        self.processed_data_folder = os.path.join(
            self.work_folder, "processed"
        )
        self.dataset_file_name = dataset_file_name

        # Data speciifics
        self.X_timeseries = X_timeseries
        self.Y_timeseries = Y_timeseries
        self.datetime_col = datetime_col
        self.columns_Y = columns_Y

        self.keras_backend = keras_backend
        self.process_complete = False

        self.process_fn = process_fn
        self.read_fn = read_fn
        self.validation_fn = validation_fn
        self.process_benchmark_fn = process_benchmark_fn

        self.keras_sequence_cls = keras_sequence_cls
        self.sequence_args = sequence_args or {}

        self.benchmark_score_path = score_path

        super().__init__(
            obj_type="datahandler", work_folder=self.work_folder, **kwargs
        )

    def obj_setup(self):
        self.processed_data_path = os.path.join(
            self.processed_data_folder, self.dataset_file_name
        )

        if os.path.exists(self.processed_data_path):
            self.process_complete = True
        else:
            self.process_data()

        dataframe = self.read_data()
        self.y_mean = dataframe[self.columns_Y].mean().item()
        self.n_features_train = len(
            dataframe.drop(self.datetime_col, axis=1).columns
        )
        self.n_features_predict = len(self.columns_Y)

        if self.keras_sequence_cls is None:
            from alquitable.generator import DataGenerator

            self.keras_sequence_cls = DataGenerator

    def get_validation_dataframe(self):
        return self.validation_fn(
            copy.deepcopy(self.read_data()), columns_Y=self.columns_Y
        )

    def process_data(self, **kwargs):
        if self.process_complete:
            return
        self.process_fn(y_columns=self.columns_Y, **kwargs)
        self.process_complete = True

    def process_benchmark(self):
        self.process_benchmark_fn(
            self.benchmark_data(),
            self.benchmark_data(return_validation_dataset_Y=True),
            self.name,
        )

    def read_data(self, **kwargs) -> pd.DataFrame:
        return self.read_fn(self.processed_data_path, **kwargs)

    def validation_data(self, **kwargs):
        return self.sequence_ravel(
            copy.deepcopy(self.get_validation_dataframe()), frac=1, **kwargs
        )

    def benchmark_data(self, return_validation_dataset_Y=False, **kwargs):
        # TODO: wtf, too specific for this case....
        (
            validation_dataset_X,
            validation_dataset_Y,
            _,
            _,
        ) = self.validation_data(skiping_step=24, train_features_folga=24)
        if return_validation_dataset_Y:
            return validation_dataset_Y

        alloc_dict = {
            "UpwardUsedSecondaryReserveEnergy": "SecondaryReserveAllocationAUpward",
            "DownwardUsedSecondaryReserveEnergy": "SecondaryReserveAllocationADownward",
        }
        alloc_column = []
        for y in self.columns_Y:
            allo = alloc_dict[y]
            alloc_column.append(allo)

        validation_benchmark = (
            self.get_validation_dataframe()[alloc_column]
            .iloc[self.X_timeseries : -self.Y_timeseries]
            .values.reshape(validation_dataset_Y.shape)
        )

        return validation_benchmark

    def keras_sequence(self, **kwargs) -> keras.utils.Sequence:
        return self.keras_sequence_cls(**kwargs)

    def sequence_ravel(self, dataframe_to_use, frac=1, **kwargs):
        kwargs_to_use = copy.deepcopy(self.sequence_args)
        kwargs_to_use.update(kwargs)
        data_generator = self.keras_sequence_cls(
            dataset=copy.deepcopy(dataframe_to_use),
            time_moving_window_size_X=self.X_timeseries,
            time_moving_window_size_Y=self.Y_timeseries,
            y_columns=self.columns_Y,
            **kwargs_to_use,
        )

        X, Y = [], []
        for x, y in data_generator:
            X.append(x)
            Y.append(y)
        X = np.array(X)
        Y = np.array(Y)
        train_len = math.ceil(frac * len(X))
        test_len = len(X) - train_len

        train_dataset_X = X[:train_len]
        test_dataset_X = X[train_len : train_len + test_len]

        train_dataset_Y = Y[:train_len]
        test_dataset_Y = Y[train_len : train_len + test_len]

        return (
            train_dataset_X,
            train_dataset_Y,
            test_dataset_X,
            test_dataset_Y,
        )

    def training_data(self, frac=1, **kwargs):
        return self.sequence_ravel(self.read_data(), frac=frac, **kwargs)



def ExperimentFactory(

    data_handlers=None,

target_variables=None,    

    previous_experiment_dict=None,
    **kwargs,
):
    previous_experiment_dict = previous_experiment_dict or {}
    experiment_dict = {}
    for target_variable in target_variables:
        data_manager = data_handlers[target_variable]
        previous_experiment = previous_experiment_dict.get(target_variable, None)
        experiment_handles = ExperimentHandler(
                                target_variable=target_variable,
                                data_manager=data_manager,
                                previous_experiment=previous_experiment,
                                **kwargs,
                                )

        experiment_dict[target_variable] = experiment_handles

    return experiment_dict