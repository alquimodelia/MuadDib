import copy
import math
import os

import keras_core
import numpy as np
import pandas as pd

from muaddib.shaihulud_utils import get_target_dict

DATA_FOLDER = os.getenv("DATA_FOLDER", None)
PROCESSED_DATA_PATH = os.path.join(DATA_FOLDER, "processed")
RAW_DATA_PATH = os.path.join(DATA_FOLDER, "raw")

X_TIMESERIES = os.getenv("X_TIMESERIES", 168)
Y_TIMESERIES = os.getenv("Y_TIMESERIES", 24)

TARGET_VARIABLE = os.getenv("TARGET_VARIABLE")
benchmark_data_folder = os.path.join(DATA_FOLDER, "benchmark", TARGET_VARIABLE)
score_path = os.path.join(benchmark_data_folder, "benchmark.json")


class DatasetManager:
    def __init__(
        self,
        work_folder=DATA_FOLDER,
        raw_data_folder=RAW_DATA_PATH,
        processed_data_folder=PROCESSED_DATA_PATH,
        dataset_file_name="dados_2014-2022.csv",  # TODO: change to something more general
        name=None,
        # Data speciifics
        X_timeseries=X_TIMESERIES,
        Y_timeseries=Y_TIMESERIES,
        columns_Y=None,  # List
        datetime_col="datetime",
        keras_backend="torch",
        process_fn=None,
        read_fn=None,
        validation_fn=None,
        process_benchmark_fn=None,
        keras_sequence_cls=None,
        sequence_args=None,
    ):
        self.name = name
        self.work_folder = work_folder
        self.raw_data_folder = raw_data_folder
        self.processed_data_folder = processed_data_folder
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

        self.setup()

    def setup(self):
        self.processed_data_path = os.path.join(
            self.processed_data_folder, self.dataset_file_name
        )

        if os.path.exists(self.processed_data_path):
            self.process_complete = True
        else:
            self.process_data()

        self.dataframe = self.read_data()
        self.n_features_train = len(
            self.dataframe.drop(self.datetime_col, axis=1).columns
        )
        self.n_features_predict = len(self.columns_Y)

        if self.keras_sequence_cls is None:
            from alquitable.generator import DataGenerator

            self.keras_sequence_cls = DataGenerator
        self.validation_dataframe = self.validation_fn(
            copy.deepcopy(self.dataframe), columns_Y=self.columns_Y
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
            copy.deepcopy(self.validation_dataframe), frac=1, **kwargs
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
            self.validation_dataframe[alloc_column]
            .iloc[self.X_timeseries : -self.Y_timeseries]
            .values.reshape(validation_dataset_Y.shape)
        )

        return validation_benchmark

    def keras_sequence(self, **kwargs) -> keras_core.utils.Sequence:
        return self.keras_sequence_cls(**kwargs)

    def sequence_ravel(self, dataframe_to_use, frac=1, **kwargs):
        kwargs_to_use = copy.deepcopy(self.sequence_args)
        kwargs_to_use.update(kwargs)
        data_generator = self.keras_sequence_cls(
            dataset=copy.deepcopy(dataframe_to_use),
            time_moving_window_size_X=self.X_timeseries,
            time_moving_window_size_Y=self.Y_timeseries,
            y_columns=self.columns_Y,
            **kwargs_to_use
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
        return self.sequence_ravel(self.dataframe, frac=frac, **kwargs)


def DatasetFactory(target_variable=None, name=None, **kwargs):
    target_variable = target_variable or TARGET_VARIABLE
    # target_variable = "Upward;Downward"
    # target_variable = "Upward|Downward"
    # target_variable = "Upward;Downward|Tender"
    # target_variable = "Upward;Downward|Tender|Upward;Downward"
    # target_variable = "Upward;Downward|Upward;Downward"
    # target_variable = "Upward;Downward|Tender|Upward;Downward"
    final_targets = get_target_dict(target_variable)
    dataset_manager_dict = {}
    for tag_name, targets in final_targets.items():
        targets_to_use = targets.copy()
        if not isinstance(targets_to_use, list):
            targets_to_use = [targets_to_use]
        name_to_use = name
        if name is None:
            name_to_use = tag_name
        dataman = DatasetManager(
            columns_Y=targets_to_use, name=name_to_use, **kwargs
        )
        dataset_manager_dict[tag_name] = dataman
    return dataset_manager_dict
