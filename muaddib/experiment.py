# ShaiHulud
import glob
import inspect
import itertools
import json
import os

import mlflow
from mlflow import MlflowClient

client = MlflowClient(tracking_uri="http://127.0.0.1:8080")

# TODO: make some lightweight version of the object, like just the case.__dict__ to get str values out of it


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
        validation_fn=None,
        visualize_report_fn=None,
        keras_backend="torch",
        benchmark_score_file=None,
    ):
        callbacks = callbacks or []
        metrics = metrics or ["root_mean_squared_error"]

        self.name = name
        self.work_folder = work_folder
        self.epochs = epochs
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.callbacks = callbacks
        self.loss = loss
        self.metrics = metrics
        self.train_fn = train_fn
        self.validation_fn = validation_fn
        self.visualize_report_fn = visualize_report_fn
        self.keras_backend = keras_backend
        self.complete = False
        self.predict_score = {}
        self.benchmark_score_file = benchmark_score_file
        self.worthy_models = None

    def set_predict_score(self, new_predict_score):
        for key in new_predict_score.keys():
            value = new_predict_score[key]
            if not isinstance(value, list):
                value = [value]
            if key not in self.predict_score:
                self.predict_score[key] = value
            else:
                self.predict_score[key] += value


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
        experiment_name="",
        previous_benchmark=None,
        **kwargs,
    ):
        self.case_name = case_name
        self.model_name = model_name

        self.model = model
        self.freq_saves = freq_saves
        self.model_types = model_types
        self.model_conf_name = model_conf_name
        self.experiment_name = experiment_name

        self.best_result = None
        self.previous_benchmark = previous_benchmark
        self.worthy = False
        super().__init__(work_folder=work_folder, **kwargs)
        self.setup()

    def setup(
        self,
    ):
        if self.name is None:
            self.name = self.model_name

        if self.case_name:
            self.name = f"{self.name}_{self.case_name}"

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
        self.list_freq_saves = list_freq_saves
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

    def set_mlflow(self):
        # Sets the current active experiment to the "Apple_Models" experiment and
        # returns the Experiment metadata
        # experiment = mlflow.set_experiment(self.experiment_name)
        mlflow.set_experiment(self.experiment_name)

        # Define a run name for this iteration of training.
        # If this is not set, a unique name will be auto-generated for your run.

    def train_model(self):
        if self.complete:
            return
        # if os.getenv("MLFLOW_STATE")=="on":
        # if True:

        #     from mlflow import MlflowClient
        #     client = MlflowClient(tracking_uri="http://127.0.0.1:8080")

        # Start a new MLflow run
        mlflow.set_tracking_uri("http://127.0.0.1:8080")

        self.set_mlflow()
        run = mlflow.start_run(run_name=self.name)
        mlflow_callback = mlflow.keras_core.MLflowCallback(run)
        if mlflow_callback not in self.fit_args["callbacks"]:
            self.fit_args["callbacks"].append(mlflow_callback)

        self.train_fn(
            self.model,
            fit_args=self.fit_args,
            compile_args=self.compile_args,
            model_name=self.name,
        )
        # End the MLflow run
        mlflow.end_run()

    def validate_model(self):
        if len(self.list_freq_saves) == 0:
            self.setup()
        for freq_save in self.list_freq_saves:
            epoch = int(
                os.path.basename(freq_save).replace(self.model_types, "")
            )

            prediction_path = freq_save.replace(
                "freq_saves", "freq_predictions"
            ).replace(".keras", ".npz")
            score_path = freq_save.replace(
                "freq_saves", "freq_predictions"
            ).replace(".keras", ".json")

            bool_prediction = os.path.exists(prediction_path)
            bool_score = os.path.exists(score_path)
            validation_done = bool_prediction & bool_score
            # TODO: change this to a way to do a new validation if the scores change?
            # A variable input args in the command
            # if true does the score agina  if false just does the fihures
            predict_score = None
            if not validation_done:
                predict_score = self.validation_fn(
                    model_path=freq_save, model_name=self.name, epoch=epoch
                )
            else:
                if bool_score:
                    with open(score_path) as f:
                        predict_score = json.load(f)
            if predict_score:
                if self.case_name:
                    predict_score["case"] = self.case_name
            self.set_predict_score(predict_score)
        if not self.complete:
            return
        self.write_report()

    def write_report(self):
        import pandas as pd

        # TODO: change this path thingys
        case_report_path = self.case_work_path.replace(
            "experiments", "reports"
        )
        os.makedirs(case_report_path, exist_ok=True)

        case_results = pd.DataFrame(self.predict_score)

        COLUMN_TO_SORT_BY = "bscore"
        ascending_to_sort = False
        # All results, ordered by epoch
        case_results.sort_values(by="epoch", inplace=True)
        self.best_result = max(case_results[COLUMN_TO_SORT_BY])
        if self.previous_benchmark:
            self.worthy = self.best_result >= self.previous_benchmark
        path_schema_csv = os.path.join(case_report_path, "case_results.csv")
        path_schema_tex = os.path.join(case_report_path, "case_results.tex")
        case_results.sort_values(
            by=COLUMN_TO_SORT_BY, ascending=ascending_to_sort, inplace=True
        )
        if len(case_results) > 10:
            case_results_sort = pd.concat(
                [case_results.head(), case_results.tail()]
            )
            case_results_sort.to_csv(path_schema_csv, index=False)
            case_results_sort.to_latex(
                path_schema_tex, escape=False, index=False, float_format="%.2f"
            )
            path_schema_csv = os.path.join(
                case_report_path, "case_results_complete.csv"
            )
            path_schema_tex = os.path.join(
                case_report_path, "case_results_complete.tex"
            )

        case_results.to_csv(path_schema_csv, index=False)
        case_results.to_latex(
            path_schema_tex, escape=False, index=False, float_format="%.2f"
        )


class Experiment(SpiceEyes):
    def __init__(
        self,
        name,
        work_folder,
        models_dict=None,
        description="",
        experiment_tags=None,
        previous_experiment=None,
        previous_cases=None,
        **kwargs,
    ):
        self.models_dict = models_dict
        self.previous_experiment = previous_experiment
        self.previous_cases = previous_cases
        self.best_result = None

        super().__init__(name=name, work_folder=work_folder, **kwargs)
        self.description = description
        self.set_tags(experiment_tags)
        # self.set_mlflow_experiment()
        # self.setup()
        # self.experiment_configuration(models_dict)

    def set_tags(self, experiment_tags=None):
        experiment_tags = experiment_tags or {}
        default_tags = {
            "project_name": os.getenv("PROJECT_NAME", "muaddib_project"),
            "mlflow.note.content": self.description,
        }

        default_tags.update(experiment_tags)
        self.experiment_tags = default_tags

    def set_mlflow_experiment(self):
        mlflow_experiment = client.search_experiments(
            filter_string="name = '{0}'".format(self.name)
        )
        if len(mlflow_experiment) > 0:
            self.mlflow_experiment = mlflow_experiment[0]
        else:
            self.mlflow_experiment = client.create_experiment(
                name=self.name, tags=self.experiment_tags
            )

    def setup(
        self,
    ):
        self.case_work_path = os.path.join(self.work_folder, self.name)

        os.makedirs(self.case_work_path, exist_ok=True)
        if self.previous_experiment:
            self.previous_cases = self.previous_experiment.worthy_cases

        self.set_mlflow_experiment()
        self.experiment_configuration(self.models_dict)

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
        previous_benchmark=None,
        name=None,
    ):
        case_list = []
        metrics = metrics or self.metrics
        callbacks = callbacks or self.callbacks
        epochs = epochs or self.epochs

        # Define a dictionary with the variables and their default values
        variables = {
            "optimizer": optimizer or self.optimizer,
            "loss": loss or self.loss,
            "batch_size": batch_size or self.batch_size,
        }

        vars_lists = []
        for key, value in variables.items():
            # Check if the value is a list
            if isinstance(value, list):
                # This give me a list of which keys are cases
                vars_lists.append(key)
            else:
                variables[key] = [value]

        # Extract the lists from the dictionary
        lists = list(variables.values())

        # Use itertools.product to get all combinations
        for combination in itertools.product(*lists):
            case_name = ""
            # Zip the keys with the combination
            args = {
                key: value for key, value in zip(variables.keys(), combination)
            }
            # get the case name
            for key, value in args.items():
                if key in vars_lists:
                    if isinstance(value, int):
                        key_name = value
                    elif isinstance(value, str):
                        key_name = value
                    else:
                        key_name = value.name
                        # Just the 1st letter of each word
                        key_name = "".join([f[0] for f in key_name.split("_")])

                    case_name += f"{key_name}_"
            if case_name.endswith("_"):
                case_name = case_name[:-1]
            case_obj = Case(
                **args,
                metrics=metrics,
                callbacks=callbacks,
                model_name=model_name,
                case_name=case_name,
                epochs=epochs,
                model=model,
                work_folder=self.case_work_path,
                train_fn=self.train_fn,
                validation_fn=self.validation_fn,
                experiment_name=self.name,
                previous_benchmark=previous_benchmark,
                name=name,
            )
            case_list.append(case_obj)
            self.complete = self.complete & case_obj.complete
            self.set_predict_score(case_obj.predict_score)

        # # Iterate over the dictionary
        # for key, value in variables.items():
        #     # Check if the value is a list
        #     if isinstance(value, list):
        #         # If the value is a list, iterate over it
        #         for v in value:
        #             if isinstance(v, int):
        #                 case_name = v
        #             elif isinstance(v, str):
        #                 case_name=v
        #             else:
        #                 case_name = v.name
        #                 # Just the 1st letter of each word
        #                 case_name = "".join(
        #                     [f[0] for f in case_name.split("_")]
        #                 )
        #             # Create a Case object for each entry in the list
        #             case_obj = Case(
        #                 **{key: v},
        #                 **{k: v for k, v in variables.items() if k != key},
        #                 metrics=metrics,
        #                 callbacks=callbacks,
        #                 model_name=model_name,
        #                 case_name=case_name,
        #                 epochs=epochs,
        #                 model=model,
        #                 name=None,
        #                 work_folder=self.case_work_path,
        #                 train_fn=self.train_fn,
        #                 validation_fn=self.validation_fn,
        #                 experiment_name=self.name,
        #             )
        #             case_list.append(case_obj)
        #             self.complete = self.complete & case_obj.complete
        #             self.set_predict_score(case_obj.predict_score)
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
                validation_fn=self.validation_fn,
                experiment_name=self.name,
                previous_benchmark=previous_benchmark,
            )
            case_list.append(case_obj)
            self.complete = self.complete & case_obj.complete
            self.set_predict_score(case_obj.predict_score)

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
        # TODO: REMOVE list in future and use only the dict base thingy, maybe change name
        models_dict = models_dict or self.models_dict
        self.conf = []
        self.study_cases = {}
        if self.previous_cases:
            for previous_case_obj in self.previous_cases:
                variables = {
                    "optimizer": isinstance(self.optimizer, list),
                    "loss": isinstance(self.loss, list),
                    "batch_size": isinstance(self.batch_size, list),
                }
                if variables["optimizer"]:
                    opt = self.optimizer
                else:
                    opt = previous_case_obj.optimizer

                if variables["loss"]:
                    lll = self.loss
                else:
                    lll = previous_case_obj.loss
                if variables["batch_size"]:
                    bbss = self.batch_size
                else:
                    bbss = previous_case_obj.batch_size
                # name I want previous_case_obj.name
                previous_benchmark = previous_case_obj.best_result
                if self.previous_experiment.best_result:
                    previous_benchmark = self.previous_experiment.best_result
                case_conf_list = self.case_configuration(
                    model_name=previous_case_obj.model_name,
                    model=previous_case_obj.model,
                    optimizer=opt,
                    loss=lll,
                    batch_size=bbss,
                    metrics=previous_case_obj.metrics,
                    epochs=self.epochs,
                    callbacks=self.callbacks,
                    # weight=previous_case_obj.weight
                    previous_benchmark=previous_benchmark,
                    name=previous_case_obj.name,
                    **kwargs,
                )
                for case_obj in case_conf_list:
                    self.conf.append(case_obj)
                    self.study_cases[case_obj.name] = case_obj
        else:
            for model_name, model in models_dict.items():
                case_conf_list = self.case_configuration(
                    model_name=model_name, model=model, **kwargs
                )
                for case_obj in case_conf_list:
                    self.conf.append(case_obj)
                    self.study_cases[case_obj.name] = case_obj
        return self.conf

    def validate_experiment(self):
        for case_obj in self.conf:
            self.set_predict_score(case_obj.predict_score)
            if not self.best_result:
                self.best_result = case_obj.best_result
            else:
                self.best_result = max(
                    [case_obj.best_result, self.best_result]
                )

        self.write_report()

    # TODO: change all this mess of report writing
    # TODO: outsource this somewhere else, its too specific for my thesis case.
    def write_report(self):
        print(f"writin report for {self.name}")
        import pandas as pd

        # TODO: change this path thingys
        case_report_path = self.case_work_path.replace(
            "experiments", "reports"
        )
        os.makedirs(case_report_path, exist_ok=True)

        case_results = pd.DataFrame(self.predict_score)

        COLUMN_TO_SORT_BY = "bscore"
        ascending_to_sort = False
        # All results, ordered by epoch
        path_schema_csv = os.path.join(
            case_report_path, "experiment_results.csv"
        )
        path_schema_tex = os.path.join(
            case_report_path, "experiment_results.tex"
        )
        case_results.sort_values(
            by=COLUMN_TO_SORT_BY, ascending=ascending_to_sort, inplace=True
        )
        if len(case_results) > 10:
            case_results_sort = pd.concat(
                [case_results.head(), case_results.tail()]
            )
            case_results_sort.to_csv(path_schema_csv, index=False)
            case_results_sort.to_latex(
                path_schema_tex, escape=False, index=False, float_format="%.2f"
            )
            path_schema_csv = os.path.join(
                case_report_path, "experiment_results_complete.csv"
            )
            path_schema_tex = os.path.join(
                case_report_path, "experiment_results_complete.tex"
            )

        case_results.to_csv(path_schema_csv, index=False)
        case_results.to_latex(
            path_schema_tex, escape=False, index=False, float_format="%.2f"
        )

        # Sort the DataFrame by "optimal percentage" column in descending order
        case_results.sort_values(
            by=COLUMN_TO_SORT_BY, ascending=False, inplace=True
        )

        # Get the best score for each unique value in the "name" column
        best_scores = case_results.loc[
            case_results.groupby("name").idxmax()[COLUMN_TO_SORT_BY]
        ].sort_values(by=COLUMN_TO_SORT_BY, ascending=ascending_to_sort)

        path_schema_tex = os.path.join(
            case_report_path, "experiment_results_best_of_each.tex"
        )

        best_scores.to_latex(
            path_schema_tex, escape=False, index=False, float_format="%.2f"
        )

        # Get the 2nd and 3rd best scores for each unique value in the "name" column
        if ascending_to_sort is False:
            second_third_best_scores = case_results.groupby("name").apply(
                lambda x: x.nlargest(3, COLUMN_TO_SORT_BY)
            )
        else:
            second_third_best_scores = case_results.groupby("name").apply(
                lambda x: x.nsmallest(3, COLUMN_TO_SORT_BY)
            )

        second_third_best_scores.sort_values(
            by=COLUMN_TO_SORT_BY, ascending=ascending_to_sort, inplace=True
        )

        path_schema_tex = os.path.join(
            case_report_path, "experiment_results_best3.tex"
        )

        second_third_best_scores.to_latex(
            path_schema_tex, escape=False, index=False, float_format="%.2f"
        )

        no_missin_scores = case_results[case_results["bscore m"] >= 0]
        no_missin_scores = no_missin_scores[no_missin_scores["bscore s"] >= 0]
        no_missin_scores = no_missin_scores.dropna().sort_values(
            by="bscore", ascending=False
        )

        path_schema_tex = os.path.join(
            case_report_path, "experiment_results_best_10_under_benchmark.tex"
        )

        no_missin_scores.head(10).to_latex(
            path_schema_tex, escape=False, index=False, float_format="%.2f"
        )

        path_schema_tex = os.path.join(
            case_report_path, "experiment_results_best_under_benchmark.tex"
        )

        no_missin_scores.to_latex(
            path_schema_tex, escape=False, index=False, float_format="%.2f"
        )
        unique_values_list = no_missin_scores["name"].unique().tolist()
        print("----------------------------------------------------")
        print("Worthy models are: ", unique_values_list)
        self.worthy_models = unique_values_list
        self.worthy_cases = [
            self.study_cases[f]
            for f in unique_values_list
            # if self.study_cases[f].worthy
        ]

        if ascending_to_sort is False:
            no_missin_scores = no_missin_scores.groupby("name").apply(
                lambda x: x.nlargest(3, COLUMN_TO_SORT_BY)
            )
        else:
            no_missin_scores = no_missin_scores.groupby("name").apply(
                lambda x: x.nsmallest(3, COLUMN_TO_SORT_BY)
            )

        no_missin_scores.sort_values(
            by="bscore", ascending=False, inplace=True
        )

        path_schema_tex = os.path.join(
            case_report_path, "experiment_results_best3_under_benchmark.tex"
        )

        no_missin_scores.to_latex(
            path_schema_tex, escape=False, index=False, float_format="%.2f"
        )

        # TODO: make a plot with the real data and the best predictions
        # the plot is the one year, one month, one week, one day
        # maybe the best and the worse of each
        benchmark_prediction_file = self.benchmark_score_file.replace(
            ".json", "npz"
        )

    def visualize_report(self):
        # TODO: change this path thingys
        folder_figures = self.case_work_path.replace("experiments", "reports")
        # TODO: get this from an envi
        metrics_to_check = None
        benchmark_score = {}

        if os.path.exists(self.benchmark_score_file):
            with open(self.benchmark_score_file) as f:
                benchmark_score = json.load(f)

        figure_name = "experiment_results.png"
        self.visualize_report_fn(
            self.predict_score,
            metrics_to_check=metrics_to_check,
            benchmark_score=benchmark_score,
            folder_figures=folder_figures,
            figure_name=figure_name,
        )

        self.visualize_report_fn(
            self.predict_score,
            metrics_to_check=metrics_to_check,
            benchmark_score=benchmark_score,
            folder_figures=folder_figures,
            figure_name=figure_name,
            limit_by="outliers",
        )
        self.visualize_report_fn(
            self.predict_score,
            metrics_to_check=metrics_to_check,
            benchmark_score=benchmark_score,
            folder_figures=folder_figures,
            figure_name=figure_name,
            limit_by="benchmark",
        )

        metrics_to_check = ["alloc missing", "alloc surplus"]
        figure_name = "experiment_results_redux.png"
        figure_size = (20, 10)
        self.visualize_report_fn(
            self.predict_score,
            metrics_to_check=metrics_to_check,
            benchmark_score=benchmark_score,
            folder_figures=folder_figures,
            figure_name=figure_name,
            figsize=figure_size,
        )

        self.visualize_report_fn(
            self.predict_score,
            metrics_to_check=metrics_to_check,
            benchmark_score=benchmark_score,
            folder_figures=folder_figures,
            figure_name=figure_name,
            limit_by="outliers",
            figsize=figure_size,
        )
        self.visualize_report_fn(
            self.predict_score,
            metrics_to_check=metrics_to_check,
            benchmark_score=benchmark_score,
            folder_figures=folder_figures,
            figure_name=figure_name,
            limit_by="benchmark",
            figsize=figure_size,
        )
