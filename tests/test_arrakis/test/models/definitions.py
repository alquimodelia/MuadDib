import itertools
import os

import forecat  # import CNNArch, DenseArch, LSTMArch, UNETArch,EncoderDecoder
import keras

MODELS_FOLDER = os.getenv("MODELS_FOLDER", None)


X_TIMESERIES = os.getenv("X_TIMESERIES", 168)
Y_TIMESERIES = os.getenv("Y_TIMESERIES", 24)

TARGET_VARIABLE = os.getenv("TARGET_VARIABLE")

TRAIN_FRACTION = float(os.getenv("TRAIN_FRACTION", 1))


epocas = 200
x_timesteps = 168
y_timesteps = 24
frac = 1
train_features_folga = 24
skiping_step = 1
keep_y_on_x = True
backed = os.getenv("KERAS_BACKEND", "tensorflow")
struct_name = f"linear_models_epocs_{backed}_mirror"

columns_Y = ["UpwardUsedSecondaryReserveEnergy"]
alloc_column = ["SecondaryReserveAllocationAUpward"]
y_columns = columns_Y

get_dataset_args = {
    "y_columns": columns_Y,
    "time_moving_window_size_X": x_timesteps,
    "time_moving_window_size_Y": y_timesteps,
    "frac": frac,
    "keep_y_on_x": keep_y_on_x,
    "train_features_folga": train_features_folga,
    "skiping_step": skiping_step,
    # "time_cols":time_cols,
    "alloc_column": alloc_column,
}

models_strutres = {
    "StackedCNN": {
        "arch": "CNNArch",
        "architecture_args": {"block_repetition": 2},
    },
    "StackedDense": {
        "arch": "DenseArch",
        "architecture_args": {"block_repetition": 2},
    },
    "VanillaCNN": {
        "arch": "CNNArch",
    },
    "VanillaDense": {"arch": "DenseArch"},
    "VanillaLSTM": {"arch": "LSTMArch"},
    "StackedLSTMA": {
        "arch": "LSTMArch",
        "architecture_args": {"block_repetition": 2},
    },
    "UNET": {"arch": "UNETArch"},
    "EncoderDecoder": {"arch": "EncoderDecoder"},
}
x_timesteps = 168
y_timesteps = 24
num_features_to_train = 18
num_classes = 1
activation_middle = "relu"
activation_end = "relu"


input_args = {
    "x_timesteps": x_timesteps,
    "y_timesteps": y_timesteps,
    "num_features_to_train": num_features_to_train,
    "num_classes": num_classes,
    "activation_middle": activation_end,
    "activation_end": activation_middle,
}


def get_models_dict(
    models_strutres=models_strutres,
    architecture_args={},
    input_args=input_args,
):
    models_dict = {}

    for model_name in models_strutres:
        model_def_path = os.path.join(MODELS_FOLDER, f"{model_name}.json")
        if not os.path.exists(model_def_path):
            # Model
            model_conf = models_strutres[model_name]
            model_conf_name = model_conf["arch"]
            model_conf_arch = getattr(forecat, model_conf_name)

            architecture_args_to_use = architecture_args.copy()
            architecture_args_to_use.update(
                model_conf.get("architecture_args", {})
            )
            architecture_args_to_use.update({"name": model_name})
            forearch = model_conf_arch(**input_args)
            models_dict[model_name] = forearch.architecture(
                **architecture_args_to_use
            )
            model_json = models_dict[model_name].to_json()
            with open(model_def_path, "w") as outfile:
                outfile.write(model_json)
        else:
            with open(model_def_path, "r") as json_file:
                loaded_model_json = json_file.read()

            models_dict[model_name] = keras.models.model_from_json(
                loaded_model_json
            )

    return models_dict


x_timesteps = 168
y_timesteps = 24
num_features_to_train = 18
num_classes = 1
activation_middle = "relu"
activation_end = "relu"
activation_list = [f for f in dir(keras.activations) if "__" not in f]
activation_list = [
    f for f in activation_list if f not in ["deserialize", "get", "serialize"]
]

input_args = {
    "x_timesteps": x_timesteps,
    "y_timesteps": y_timesteps,
    "num_features_to_train": num_features_to_train,
    "num_classes": num_classes,
    "activation_middle": activation_list,
    "activation_end": activation_list,
}


def get_models_on_experiments(
    models_to_use,
    architecture_args={},
    input_args=input_args,
):
    models_dict = {}
    models_to_use = [f.split("_")[0] for f in models_to_use]
    vars_lists = []
    for key, value in input_args.items():
        # Check if the value is a list
        if isinstance(value, list):
            # This give me a list of which keys are cases
            vars_lists.append(key)
        else:
            input_args[key] = [value]
        # Extract the lists from the dictionary
    lists = list(input_args.values())
    # Use itertools.product to get all combinations
    for combination in itertools.product(*lists):
        case_name = ""
        # Zip the keys with the combination
        args = {
            key: value for key, value in zip(input_args.keys(), combination)
        }
        x_timesteps = args["x_timesteps"]
        y_timesteps = args["y_timesteps"]
        X_in_vars = "x_timesteps" in vars_lists
        Y_in_vars = "y_timesteps" in vars_lists
        time_in_vars = X_in_vars or Y_in_vars

        activation_middle = args["activation_middle"]
        activation_end = args["activation_end"]
        activation_middle_in_vars = "activation_middle" in vars_lists
        activation_end_in_vars = "activation_end" in vars_lists
        activation_in_vars = (
            activation_middle_in_vars or activation_end_in_vars
        )

        if args["x_timesteps"] < args["y_timesteps"]:
            continue
        if time_in_vars:
            case_name += f"X{x_timesteps}_Y{y_timesteps}_"
        if activation_in_vars:
            case_name += f"{activation_middle}_{activation_end}_"
        if case_name.endswith("_"):
            case_name = case_name[:-1]

        for model_to_use in models_to_use:
            model_to_fetch = model_to_use.split("_")[0]
            model_name = model_to_use + "_" + case_name

            model_def_path = os.path.join(MODELS_FOLDER, f"{model_name}.json")
            if not os.path.exists(model_def_path):
                # Model
                model_conf = models_strutres[model_to_fetch]
                model_conf_name = model_conf["arch"]
                model_conf_arch = getattr(forecat, model_conf_name)

                architecture_args_to_use = architecture_args.copy()
                architecture_args_to_use.update(
                    model_conf.get("architecture_args", {})
                )
                architecture_args_to_use.update({"name": model_name})
                forearch = model_conf_arch(**args)
                models_dict[model_name] = forearch.architecture(
                    **architecture_args_to_use
                )
                model_json = models_dict[model_name].to_json()
                with open(model_def_path, "w") as outfile:
                    outfile.write(model_json)
            else:
                with open(model_def_path, "r") as json_file:
                    loaded_model_json = json_file.read()

                models_dict[model_name] = keras.models.model_from_json(
                    loaded_model_json
                )
    return models_dict


ALL_MODELS_DICT = get_models_dict()
