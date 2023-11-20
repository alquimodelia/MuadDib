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
        keras_sequence_cls=None,
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

        self.keras_sequence_cls = keras_sequence_cls

    def start_data_manager(self):
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

    def process_data(self, **kwargs):
        if self.process_complete:
            return
        self.process_fn(**kwargs)
        self.process_complete = True

    def read_data(self, **kwargs) -> pd.DataFrame:
        self.read_fn(self.processed_data_path, **kwargs)

    def keras_sequence(self, **kwargs) -> keras_core.utils.Sequence:
        return self.keras_sequence_cls(**kwargs)

    def sequence_ravel(self, frac=1, **kwargs):
        data_generator = self.keras_sequence_cls(
            dataset=copy.deepcopy(self.dataframe),
            time_moving_window_size_X=self.X_timeseries,
            time_moving_window_size_Y=self.Y_timeseries,
            y_columns=self.columns_Y,
            **kwargs
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


def DatasetFactory(target_variable=None, name=None, **kwargs):
    target_variable = target_variable or os.getenv("TARGET_VARIABLE")
    # target_variable = "Upward;Downward"
    # target_variable = "Upward|Downward"
    # target_variable = "Upward;Downward|Tender"
    # target_variable = "Upward;Downward|Tender|Upward;Downward"
    # target_variable = "Upward;Downward|Upward;Downward"
    # target_variable = "Upward;Downward|Tender|Upward;Downward"
    final_targets = get_target_dict(target_variable)
    dataset_manager_dict = {}
    for tag_name, targets in final_targets.items():
        if not isinstance(targets, list):
            targets = [targets]
        name_to_use = name
        if name is None:
            name_to_use = tag_name
        dataman = DatasetFactory(columns_Y=targets, name=name_to_use, **kwargs)
        dataset_manager_dict[tag_name] = dataman

    return dataset_manager_dict
