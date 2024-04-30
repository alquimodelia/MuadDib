import copy
import math
import os

import keras
import numpy as np
import pandas as pd

from muaddib.muaddib import ShaiHulud


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
        read_data_args=None,
        commun_steps=0,
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
        self.commun_steps = commun_steps
        self.datetime_col = datetime_col
        self.columns_Y = columns_Y

        self.keras_backend = keras_backend
        self.process_complete = False

        self.read_data_args = copy.deepcopy(read_data_args) or {}

        self.process_fn = process_fn
        self.read_fn = read_fn
        self.validation_fn = validation_fn
        self.process_benchmark_fn = process_benchmark_fn
        self.sequence_args = copy.deepcopy(sequence_args) or {}

        self.keras_sequence_cls = keras_sequence_cls
        if commun_steps > 0:
            self.sequence_args["commun_timesteps"] = commun_steps

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
        return self.read_fn(
            self.processed_data_path, **self.read_data_args, **kwargs
        )

    def validation_data(self, **kwargs):
        return self.sequence_ravel(
            copy.deepcopy(self.get_validation_dataframe()), frac=1, **kwargs
        )

    def benchmark_data(
        self,
        return_validation_dataset_Y=False,
        alloc_dict=None,
        skiping_step=24,
        train_features_folga=24,
        **kwargs,
    ):
        alloc_dict = alloc_dict or {
            "UpwardUsedSecondaryReserveEnergy": "SecondaryReserveAllocationAUpward",
            "DownwardUsedSecondaryReserveEnergy": "SecondaryReserveAllocationADownward",
        }
        # TODO: wtf, too specific for this case....
        (
            validation_dataset_X,
            validation_dataset_Y,
            _,
            _,
        ) = self.validation_data(
            skiping_step=skiping_step,
            train_features_folga=train_features_folga,
        )
        if return_validation_dataset_Y:
            return validation_dataset_Y

        alloc_column = self.columns_Y
        if isinstance(alloc_dict, dict):
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

        num_batches = len(data_generator)

        X, Y = [], []
        for i in range(num_batches):
            x, y = data_generator[i]
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
