import copy
import os
from test.models.predict_model import prediction_score, save_scores

import pandas as pd
from alquitable.generator import DataGenerator
from sklearn.impute import IterativeImputer

from muaddib.data import DatasetFactory

PROCESSED_FILE_PATH = os.getenv("PROCESSED_FILE_PATH")

RAW_FILE_PATH = os.getenv("RAW_FILE_PATH")
PROCESSED_FILE_PATH = os.getenv("PROCESSED_FILE_PATH")
target_variable = os.getenv("TARGET_VARIABLE")
PROCESSED_FILE_NAME = os.getenv("PROCESSED_FILE_NAME")
X_TIMESERIES = os.getenv("X_TIMESERIES", 168)
Y_TIMESERIES = os.getenv("Y_TIMESERIES", 24)


datetime_col = "datetime"
time_cols = [
    "hour",
    "day",
    "month",
    "year",
    "day_of_year",
    "day_of_week",
    "week_of_year",
]


def process_data_fn(y_columns, path_raw=RAW_FILE_PATH, **kwargs):
    dataset = pd.read_csv(path_raw, index_col=0)

    d = pd.to_datetime(dataset[datetime_col], format="mixed", utc=True)

    dataset["hour"] = [f.hour for f in d]
    dataset["day"] = [f.day for f in d]
    dataset["month"] = [f.month for f in d]
    dataset["year"] = [f.year for f in d]
    dataset["day_of_year"] = [f.timetuple().tm_yday for f in d]
    dataset["day_of_week"] = [f.timetuple().tm_wday for f in d]
    dataset["week_of_year"] = [f.weekofyear for f in d]

    # Make the y the 1st column
    dataset = dataset[
        y_columns + [col for col in dataset.columns if col not in y_columns]
    ]

    # make the time columns the last
    dataset = dataset[
        [col for col in dataset.columns if col not in time_cols] + time_cols
    ]

    # Photovoltaic has a lot of missing data so frist get night time to 0
    mask_dark = dataset["hour"].isin([21, 22, 23, 0, 1, 2, 3, 4])
    dataset.loc[mask_dark, "PhotovoltaicD+1DailyForecast"] = 0

    df = dataset.copy()
    df.drop("datetime", axis=1, inplace=True)
    # Sort DataFrame by DateTime index
    df.sort_index(inplace=True)

    # Perform imputation
    imputer = IterativeImputer(max_iter=1000, random_state=0)
    df_imputed = imputer.fit_transform(df)

    # Convert the result back to a DataFrame
    df_imputed = pd.DataFrame(df_imputed, columns=df.columns, index=df.index)
    df_imputed["datetime"] = dataset["datetime"]

    df_imputed.to_csv(PROCESSED_FILE_PATH)

    # Contruct the method to process the data

    # Save final processed file to PROCESSED_FILE_PATH
    return df_imputed


def read_data_fn(path):
    return pd.read_csv(path, index_col=0)


def validation_data_fn(dataset, columns_Y, years_to_use=None, **kwargs):
    if years_to_use is None:
        years_to_use = [2019, 2020, 2021, 2022]

    validation_dataset = copy.deepcopy(dataset)

    validation_dataset["date"] = pd.to_datetime(
        validation_dataset[datetime_col]
    )
    year_mask = validation_dataset["date"].dt.year.isin(years_to_use)
    validation_dataset[year_mask][columns_Y]

    validation_dataset["hour_in_year"] = (
        validation_dataset["date"].dt.hour
    ) + (24 * (validation_dataset["day_of_year"] - 1))
    mask_before = validation_dataset["date"].dt.year == min(years_to_use) - 1
    max_hour = max(validation_dataset[mask_before]["hour_in_year"])

    mask_last_hours = validation_dataset["hour_in_year"] > (
        max_hour - X_TIMESERIES
    )
    mask_before = mask_before & mask_last_hours

    mask_after = validation_dataset["date"].dt.year == max(years_to_use) + 1
    mask_after_hours = validation_dataset["hour_in_year"] < Y_TIMESERIES
    mask_after = mask_after & mask_after_hours

    mask_data = mask_before | year_mask | mask_after

    validation_dataset = copy.deepcopy(dataset[mask_data])

    return validation_dataset

    phased_out_columns = (None,)
    alloc_column = (None,)


def process_benchmark_fn(
    validation_benchmark, validation_dataset_Y, target_variable
):
    # Do benchmark
    DATA_FOLDER = os.getenv("DATA_FOLDER", "data")
    benchmark_data_folder = os.path.join(
        DATA_FOLDER, "benchmark", target_variable
    )
    score_path = os.path.join(benchmark_data_folder, "benchmark.json")
    test_path = os.path.join(benchmark_data_folder, "benchmark.npz")

    benchmark_scores = prediction_score(
        validation_dataset_Y,
        validation_benchmark,
        validation_benchmark,
        "benchmark",
    )
    new_benchmark_scores = benchmark_scores.copy()
    # TODO: handle keys to remove in some way
    for key in benchmark_scores.keys():
        if "EPEA" in key:
            new_benchmark_scores.pop(key)
        if "percentage" in key:
            new_benchmark_scores.pop(key)

    save_scores(
        validation_dataset_Y,
        validation_benchmark,
        validation_benchmark,
        test_path,
        new_benchmark_scores,
        score_path,
    )


factory_args = {
    "dataset_file_name": PROCESSED_FILE_NAME,
    "X_timeseries": X_TIMESERIES,
    "Y_timeseries": Y_TIMESERIES,
    "datetime_col": "datetime",
    "process_fn": process_data_fn,
    "read_fn": read_data_fn,
    "keras_sequence_cls": DataGenerator,
    "validation_fn": validation_data_fn,
    "process_benchmark_fn": process_benchmark_fn,
    "sequence_args": {
        "skiping_step": 1,
        "keep_y_on_x": True,
        "train_features_folga": 24,
        "time_cols": time_cols,
        "drop_cols": "datetime",
        "phased_out_columns": [
            "UpwardUsedSecondaryReserveEnergy",
            "DownwardUsedSecondaryReserveEnergy",
        ],
    },
}

ALL_DATA_MANAGERS = DatasetFactory(target_variable, **factory_args)
