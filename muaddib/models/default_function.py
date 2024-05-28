import os

import keras
import numpy as np


def keras_train_model(
    model_obj=None,
    epochs=None,
    datamanager=None,
    optimizer=None,
    loss=None,
    batch_size=None,
    weights=None,
    callbacks=None,
    training_args=None,
    **kwargs,
):
    training_args = training_args or {}
    X, Y, _, _ = datamanager.training_data(**training_args)
    keras.backend.clear_session()

    fit_args = {
        "epochs": epochs,
        "callbacks": callbacks,
        "batch_size": batch_size,
        "shuffle": True,
    }

    compile_args = {
        "optimizer": optimizer,
        "loss": loss,
    }

    if weights:
        if isinstance(weights, bool):
            # It defaults to more weight on last batch, good for timeseries
            weights = ""
        elif isinstance(weights, str):
            weights = weights.lower()

        sam = False
        if "mean" in weights:
            sample_weights = np.abs(np.array(Y) - datamanager.y_mean)
        else:
            sam = True
            sample_weights = np.arange(len(Y)) + 1

        if isinstance(weights, str):
            if "squared" in weights.lower():
                sample_weights = sample_weights**2
            if "cubic" in weights.lower():
                sample_weights = sample_weights**3
            if "inv" in weights.lower():
                sample_weights = np.max(sample_weights) / sample_weights
            if "norm" in weights.lower():
                sample_weights = sample_weights / np.max(sample_weights)

        if sam:
            sam = sample_weights.reshape(Y.shape[0], 1, 1)
            sample_weights = np.broadcast_to(sam, Y.shape)
        # sample_weights = sample_weights.flatten()
        fit_args.update({"sample_weight": sample_weights})

    model_obj.compile(**compile_args)
    history_new = model_obj.fit(X, Y, **fit_args, **kwargs)
    return history_new


def statsmodel_train_model(
    model_obj=None,
    modelfilepath=None,
    **kwargs,
):
    model_fit = model_obj.fit(**kwargs)
    os.makedirs(os.path.dirname(modelfilepath), exist_ok=True)
    model_fit.save(modelfilepath)

    return model_fit


def get_keras_predictions(model_path, X, **valdiation_args):
    keras.backend.clear_session()
    # Compile is false because we just need to predict.
    trained_model = keras.models.load_model(model_path, compile=False)
    predictions = trained_model.predict(X)
    return predictions


def get_statsmodel_predictions(model_path, datamanager):
    from statsmodels.tsa.arima.model import ARIMAResults

    x_timesteps = datamanager.X_timeseries
    y_timesteps = datamanager.Y_timeseries
    datamanager.set_validation_index()
    validation_start_index = datamanager.validation_index_start + x_timesteps
    validation_end_index = datamanager.validation_index_end - y_timesteps

    trained_model = ARIMAResults.load(model_path)

    predictions = []
    for i in range(validation_start_index, validation_end_index, 24):
        start_index = i + x_timesteps
        end_index = start_index + y_timesteps - 1
        pred = trained_model.predict(start_index, end_index)
        predictions.append(pred)
    predictions = np.expand_dims(np.array(predictions), axis=-1)

    return predictions


def inference_model(
    prediction_path,
    prediction_name,
    model_path,
    datamanager=None,
    model_type="keras",
    benchmark=True,
    bench_args=None,
    **validation_args,
):
    predictions = None
    predictions_experiment = None
    benchmark_values = None
    prediction_file_dict = None

    X, truth_data, _, _ = datamanager.validation_data(**validation_args)
    if os.path.exists(prediction_path):
        predictions_experiment = np.load(prediction_path)
        if prediction_name in predictions_experiment.keys():
            predictions = predictions_experiment[prediction_name]
        if "benchmark" in predictions_experiment.keys():
            benchmark_values = predictions_experiment["benchmark"]
        if "test" in predictions_experiment.keys():
            truth_data = predictions_experiment["test"]

    if benchmark:
        if benchmark_values is not None:
            benchmark = benchmark_values
            benchmark_values = None  # avoid duplication
        else:
            bench_args = bench_args or {}
            benchmark = datamanager.benchmark_data(**bench_args)

    if predictions is None:
        if model_type == "keras":
            predictions = get_keras_predictions(model_path, X)
        elif model_type == "statsmodel":
            predictions = get_statsmodel_predictions(model_path, datamanager)

        if predictions_experiment is None:
            prediction_file_dict = {
                "test": truth_data,
                "benchmark": benchmark,
                prediction_name: predictions,
            }
        else:
            prediction_file_dict = dict(predictions_experiment)
            prediction_file_dict[prediction_name] = predictions

        os.makedirs(os.path.dirname(prediction_path), exist_ok=True)
        np.savez_compressed(prediction_path, **prediction_file_dict)
    if prediction_file_dict is None:
        prediction_file_dict = dict(predictions_experiment)

    return prediction_file_dict


def metric_scores_default(name, predictions, truth_data, benchmark=False):
    error = truth_data - predictions
    erro_abs = np.abs(error)
    erro_abs_sum = np.nansum(erro_abs)

    mse = np.nanmean(np.square(error))
    rmse = np.sqrt(mse)

    return {
        "erro_abs_sum": [erro_abs_sum],
        "mse": [mse],
        "rmse": [rmse],
    }


def validate_model(
    prediction_file_dict, prediction_name, benchmark=True, metric_scores=None
):
    metric_scores = metric_scores or metric_scores_default

    predictions = prediction_file_dict[prediction_name]
    truth_data = prediction_file_dict["test"]
    if benchmark:
        benchmark = prediction_file_dict["benchmark"]

    predict_score = metric_scores(
        prediction_name, predictions, truth_data, benchmark
    )

    return predict_score
