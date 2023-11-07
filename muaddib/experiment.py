# ShaiHulud
import glob
import inspect
import json
import os


class SpiceEyes:
    def __init__(
        self,
        work_folder,
        name=None,
        epochs=200,
        optimizer="adam",
        batch_size=252,
        loss="mse",
        callbacks=None,
        metrics=None,
        train_fn=None,
        keras_backend="torch",
    ):
        callbacks=callbacks or [],
        metrics=metrics or ["root_mean_squared_error"],

        self.name = name
        self.work_folder = work_folder
        self.epochs = epochs
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.callbacks = callbacks
        self.loss = loss
        self.metrics = metrics
        self.train_fn = train_fn
        self.keras_backend = keras_backend
        self.complete = False


class Case(SpiceEyes):
    def __init__(
        self,
        work_folder,
        case_name="",  # Case specific
        model_name="",  # Model Name
        model=None,
        freq_saves="freq_saves",
        model_types=".keras",
        model_conf_name="model_conf.json",
        **kwargs,
    ):
        self.case_name = case_name
        self.model_name = model_name

        self.model = model
        self.freq_saves = freq_saves
        self.model_types = model_types
        self.model_conf_name = model_conf_name

        super().__init__(work_folder=work_folder, **kwargs)
        self.setup()

    def setup(
        self,
    ):
        if self.name is None:
            if self.case_name:
                self.name = f"{self.model_name}_{self.case_name}"
            else:
                self.name = f"{self.model_name}"
        self.case_work_path = os.path.join(self.work_folder, self.name)

        os.makedirs(self.case_work_path, exist_ok=True)

        self.model_keras_path = os.path.join(
            self.case_work_path, f"{self.name}.keras"
        )

        # Frequency saves
        self.case_work_frequency_path = os.path.join(
            self.case_work_path, self.freq_saves
        )
        os.makedirs(self.case_work_frequency_path, exist_ok=True)

        # Checks how many epochs were trained
        list_query = f"{self.case_work_frequency_path}/**{self.model_types}"
        list_freq_saves = glob.glob(list_query)
        last_epoch = 0
        last_epoch_path = None
        if len(list_freq_saves) > 0:
            epocs_done = [
                int(os.path.basename(f).replace(self.model_types, ""))
                for f in list_freq_saves
            ]
            last_epoch = max(epocs_done)
            last_epoch_path = f"{self.case_work_frequency_path}/{last_epoch}{self.model_types}"
        self.last_epoch_path = last_epoch_path
        self.last_epoch = last_epoch

        self.set_compile_args()
        self.set_fit_args()

    def set_compile_args(self):
        compile_args = {
            "optimizer": self.optimizer,
            "loss": self.loss,
            "metrics": self.metrics,
        }

        self.compile_args = compile_args

    def set_fit_args(self):
        epocs_to_train = self.epochs - self.last_epoch
        if epocs_to_train < 1:
            self.complete = True
            return

        callbacks = []
        if not isinstance(self.callbacks, list):
            self.callbacks = [self.callbacks]

        for callback in self.callbacks:
            callback_args = {}
            arg_names = inspect.getfullargspec(callback).args
            if "save_frequency" in arg_names:
                callback_args["save_frequency"] = 1
            if "start_epoch" in arg_names:
                callback_args["start_epoch"] = self.last_epoch
            if "model_keras_filename" in arg_names:
                frq_model_filename_sof = (
                    f"{self.case_work_frequency_path}" + "/{epoch}.keras"
                )
                callback_args["model_keras_filename"] = frq_model_filename_sof
            if "filepath" in arg_names:
                callback_args["filepath"] = self.model_keras_path
            if "model_log_filename" in arg_names:
                callback_args[
                    "model_log_filename"
                ] = self.model_keras_path.replace(".keras", ".json")
            if "logs" in arg_names:
                model_history_filename = self.model_keras_path.replace(
                    ".keras", ".json"
                )
                if os.path.exists(model_history_filename):
                    with open(model_history_filename) as f:
                        history = json.load(f)
                    callback_args["logs"] = history
            callbacks.append(callback(**callback_args))

        fit_args = {
            "epochs": epocs_to_train,
            "callbacks": callbacks,
            "batch_size": self.batch_size,
        }

        self.fit_args = fit_args

    def train_model(self):
        if self.complete:
            return
        self.train_fn(
            self.model,
            fit_args=self.fit_args,
            compile_args=self.compile_args,
            model_name=self.name,
        )


class Experiment(SpiceEyes):
    def __init__(self, name, work_folder, models_dict=None, **kwargs):
        self.models_dict = models_dict

        super().__init__(name=name, work_folder=work_folder, **kwargs)
        self.setup()
        self.experiment_configuration(models_dict)

    def setup(
        self,
    ):
        self.case_work_path = os.path.join(self.work_folder, self.name)

        os.makedirs(self.case_work_path, exist_ok=True)

    def get_compile_args(self, optimizer, loss, metrics):
        compile_args = {
            "optimizer": optimizer,
            "loss": loss,
            "metrics": metrics,
        }

        self.compile_args = compile_args
        return compile_args

    def get_fit_args(self, epochs, callbacks, batch_size):
        fit_args = {
            "epochs": epochs,
            "callbacks": callbacks,
            "batch_size": batch_size,
        }

        self.fit_args = fit_args
        return fit_args

    def case_configuration(
        self,
        model_name,
        optimizer=None,
        loss=None,
        metrics=None,
        epochs=None,
        callbacks=None,
        batch_size=None,
        model=None,
        weight=None,
    ):
        case_list = []

        # Define a dictionary with the variables and their default values
        variables = {
            "optimizer": optimizer or self.optimizer,
            "loss": loss or self.loss,
            "batch_size": batch_size or self.batch_size,
        }

        metrics = metrics or self.metrics
        callbacks = callbacks or self.callbacks
        epochs = epochs or self.epochs
        # Iterate over the dictionary
        for key, value in variables.items():
            # Check if the value is a list
            if isinstance(value, list):
                # If the value is a list, iterate over it
                for v in value:
                    if isinstance(v, int):
                        case_name = v
                    else:
                        case_name = v.name
                        # Just the 1st letter of each word
                        case_name = [f[0] for f in case_name.split("_")]
                    # Create a Case object for each entry in the list
                    case_obj = Case(
                        **{key: v},
                        **{k: v for k, v in variables.items() if k != key},
                        metrics=metrics,
                        callbacks=callbacks,
                        model_name=model_name,
                        case_name=case_name,
                        epochs=epochs,
                        model=model,
                        name=None,
                        work_folder=self.case_work_path,
                        train_fn=self.train_fn,
                    )
                    case_list.append(case_obj)
                    self.complete = self.complete & case_obj.complete
        if len(case_list) == 0:
            # If the value is not a list, create a Case object with the value
            case_obj = Case(
                **variables,
                metrics=metrics,
                callbacks=callbacks,
                epochs=epochs,
                model_name=model_name,
                model=model,
                name=None,
                work_folder=self.case_work_path,
                train_fn=self.train_fn,
            )
            case_list.append(case_obj)
            self.complete = self.complete & case_obj.complete 

        # if weight:
        #     if "delta_mean" in  weight or "both" in weight:
        #         train_dataset_Y_values = train_dataset_Y_values or train_dataset_Y
        #         samples_weights = np.abs(train_dataset_Y_values - mean)
        #         fit_args["sample_weight"] = samples_weights
        #     if "freq" in  weight or "both" in weight:
        #         freq_weights = get_freq_samples(train_dataset_labels)
        #         fit_args["sample_weight"] = freq_weights
        #     if "both" in weight:
        #         fit_args["sample_weight"] = freq_weights*samples_weights

        return case_list

    def experiment_configuration(self, models_dict=None, **kwargs):
        models_dict = models_dict or self.models_dict

        for model_name, model in models_dict.items():
            self.conf = self.case_configuration(
                model_name=model_name, model=model, **kwargs
            )
        return self.conf
