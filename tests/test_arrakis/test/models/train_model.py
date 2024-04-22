# Define yout training method(s), with the following arguments:
import ast
import os

import keras
import numpy as np


# It is used in the default train command, or you can pass it as training_fn to the Experiment class object
# Any methods you want to create for training should have the same args, and then you pass them as training_fn to the respective Case/Experiment.
def train_model(
    model_to_train=None,
    datamanager=None,
    fit_args=None,
    compile_args=None,
    model_name="experiment_model",
    weights=False,
):
    X, Y, _, _ = datamanager.training_data()
    keras.backend.clear_session()

    if isinstance(model_to_train, str):
        if os.path.exists(model_to_train):
            model_to_train = keras.models.load_model(model_to_train)
        else:
            #             model_to_train = read_model_conf(model_to_train)
            model_to_train = keras.models.model_from_json(model_to_train)
    if isinstance(fit_args, str):
        fit_args = ast.literal_eval(fit_args)
    if isinstance(compile_args, str):
        compile_args = ast.literal_eval(compile_args)

    if weights:
        sample_weights = np.abs(np.array(Y) - datamanager.y_mean)
        fit_args.update({"sample_weight": sample_weights})

    model_to_train.compile(**compile_args)

    history_new = model_to_train.fit(X, Y, **fit_args)
