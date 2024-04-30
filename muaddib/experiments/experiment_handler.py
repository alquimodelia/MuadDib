import os

import pandas as pd

from muaddib.models.model_handler import ModelHandler
from muaddib.muaddib import ShaiHulud


class ExperimentHandler(ShaiHulud):
    def __init__(
        self,
        name=None,
        target_variable=None,
        data_manager=None,
        project_manager=None,
        model_handlers=None,
        train_fn=None,
        validation_fn=None,
        result_validation_fn=None,
        validation_target=None,
        write_report_fn=None,
        previous_experiment=None,
        final_experiment=False,
        **kwargs,
    ):
        conf_file = kwargs.get("conf_file", None)
        if conf_file and os.path.exists(conf_file):
            self.load(conf_file)
        else:
            self.target_variable = target_variable
            self.project_manager = project_manager

            self.name = name or "experiment1"

            self.work_folder = os.path.join(
                project_manager.experiment_folder,
                self.target_variable,
                data_manager.name,
                self.name,
            )
            self.data_manager = data_manager

            # model_handlers = model_handlers or []
            # if not isinstance(model_handlers, list):
            #     model_handlers = [model_handlers]
            # self.model_handlers = dict()

            # for model_handler in model_handlers:
            #     self.model_handlers[model_handler.name] = model_handler

            # extra_y_timesteps = max([0, data_manager.commun_steps])
            # # model_handler_args = {
            # #     "archs": archs,
            # #     "activation_middle": activation_middle,
            # #     "activation_end": activation_end,
            # #     "X_timeseries": data_manager.X_timeseries,
            # #     "Y_timeseries": data_manager.Y_timeseries + extra_y_timesteps,
            # #     "filters": filters,
            # # }
            # # obj_setup_args = {
            # #     "model_handler_args": model_handler_args,
            # #     "previous_experiment": previous_experiment,
            # # }
            # # self.obj_setup_args = obj_setup_args
            original_kwargs = self.get_model_handler_kwargs(kwargs)
            super().__init__(
                obj_type="experiment",
                work_folder=self.work_folder,
                # obj_setup_args=obj_setup_args,
                **original_kwargs,
            )

    def get_model_handler_kwargs(self, kwargs):
        original_kwargs = {**kwargs}
        model_handler_kwargs = {}
        for model_handler_name, model_handler in ModelHandler.registry.items():
            for kwarg in kwargs.keys():
                if (
                    kwarg in model_handler.model_args
                    or kwarg in model_handler.fit_kwargs
                    or kwarg in model_handler.class_args
                ):
                    if kwarg in original_kwargs:
                        original_kwargs.pop(kwarg)
                    model_handler_kwargs[kwarg] = kwargs[kwarg]
        experiment_model_handler = ModelHandler.create_model_handlers(
            **model_handler_kwargs,
            project_manager=self.project_manager,
            datamanager=self.data_manager,
        )
        self.model_handlers = experiment_model_handler
        return original_kwargs

    def obj_setup(self, model_handler_args=None, previous_experiment=None):
        experiments = {}
        for key, val in self.model_handlers.items():
            for case_name in val.exp_cases.keys():
                experiments[case_name] = {
                    "model_handler_name": key,
                }
        self.experiments = experiments

        self.save()

    def train_experiment(self):
        if not getattr(self, "experiments", False):
            self.obj_setup()
        for exp, exp_args in self.experiments.items():
            exp_fit_args = {**exp_args}
            # TODO: clean
            if "model" in exp_fit_args:
                exp_fit_args.pop("model")
            if "model_handler_name" in exp_fit_args:
                model_handler_name = exp_fit_args.pop("model_handler_name")
            exp_fit_args = self.model_handlers[model_handler_name].exp_cases[
                exp
            ]

            self.model_handlers[model_handler_name].train_model(
                exp,
                datamanager=self.data_manager,
                **exp_fit_args,
            )

    def validate_experiment(self, **kwargs):
        exp_results_path = os.path.join(
            self.work_folder, "experiment_score.csv"
        )
        if os.path.exists(exp_results_path):
            exp_results = pd.read_csv(exp_results_path, index_col=0)
        else:
            exp_results = pd.DataFrame()
        for exp in self.experiments.keys():
            if "name" in exp_results:
                saved_exp_score = exp_results[exp_results["name"] == exp]
                num_exps = len(saved_exp_score)
            else:
                saved_exp_score = None
                num_exps = 0
            model_handler_name = self.experiments[exp]["model_handler_name"]
            experiment_epochs = self.model_handlers[model_handler_name].epochs
            if num_exps < experiment_epochs:
                # ##########
                exp_score = self.model_handlers[
                    model_handler_name
                ].validate_model(
                    exp,
                    self.data_manager,
                    old_score=saved_exp_score,
                    **kwargs,
                )
                # #################
                exp_results = pd.concat([exp_results, exp_score])
        exp_results = exp_results.drop_duplicates(["name", "epoch"])
        exp_results = exp_results.reset_index(drop=True)
        exp_results.to_csv(exp_results_path, index=False)
        return exp_results


# class KerasExperiment(ShaiHulud):
#     def __init__(
#         self,
#         name=None,
#         # Experiment
#         optimizer=None,
#         loss=None,
#         batch_size=None,
#         weights=None,
#         # ModelHandler
#         archs=None,
#         activation_middle=None,
#         activation_end=None,
#         filters=None,
#         target_variable=None,
#         data_manager=None,
#         project_manager=None,
#         model_handlers=None,
#         train_fn=None,
#         epochs=None,
#         callbacks=None,
#         validation_fn=None,
#         result_validation_fn=None,
#         validation_target=None,
#         write_report_fn=None,
#         previous_experiment=None,
#         final_experiment=False,
#         **kwargs,
#     ):
#         """
#         We want were to have batch/weights/loss/optimizer
#         Constructor method.

#         Parameters
#         ----------
#         p1 : str, optional
#             Description of the parameter, by default "whatever"
#         """
#         conf_file = kwargs.get("conf_file", None)
#         if conf_file and os.path.exists(conf_file):
#             self.load(conf_file)
#         else:
#             self.optimizer = optimizer
#             self.loss = loss
#             self.batch_size = batch_size
#             self.weights = weights

#             self.target_variable = target_variable
#             self.project_manager = project_manager

#             self.name = name or "experiment1"

#             self.work_folder = os.path.join(
#                 project_manager.experiment_folder,
#                 self.target_variable,
#                 data_manager.name,
#                 self.name,
#             )
#             self.data_manager = data_manager
#             self.train_fn = train_fn
#             self.validation_fn = validation_fn
#             self.result_validation_fn = result_validation_fn
#             self.validation_target = validation_target
#             self.write_report_fn = write_report_fn
#             self.final_experiment = final_experiment

#             self.epochs = epochs
#             self.callbacks = callbacks
#             model_handlers = model_handlers or []
#             if not isinstance(model_handlers, list):
#                 model_handlers = [model_handlers]
#             self.model_handlers = dict()

#             for model_handler in model_handlers:
#                 self.model_handlers[model_handler.name] = model_handler

#             extra_y_timesteps = max([0, data_manager.commun_steps])
#             model_handler_args = {
#                 "archs": archs,
#                 "activation_middle": activation_middle,
#                 "activation_end": activation_end,
#                 "X_timeseries": data_manager.X_timeseries,
#                 "Y_timeseries": data_manager.Y_timeseries + extra_y_timesteps,
#                 "filters": filters,
#             }
#             obj_setup_args = {
#                 "model_handler_args": model_handler_args,
#                 "previous_experiment": previous_experiment,
#             }
#             self.obj_setup_args = obj_setup_args

#             super().__init__(
#                 obj_type="experiment",
#                 work_folder=self.work_folder,
#                 obj_setup_args=obj_setup_args,
#                 **kwargs,
#             )

#     def obj_setup(self, model_handler_args=None, previous_experiment=None):
#         model_handler_args = self.obj_setup_args.get(
#             "model_handler_args", model_handler_args
#         )
#         previous_experiment = self.obj_setup_args.get(
#             "previous_experiment", previous_experiment
#         )
#         if isinstance(self.loss, AdvanceLossHandler):
#             self.loss.set_previous_loss(previous_experiment.best_exp["loss"])
#         delattr(self, "obj_setup_args")
#         if previous_experiment:
#             self.previous_case = previous_experiment.best_case
#             previous_best_model = (
#                 previous_experiment.model_handler.models_confs[
#                     previous_experiment.best_exp["model"]
#                 ].copy()
#             )
#             previous_best_model.pop("n_features_predict")
#             previous_best_model.pop("n_features_train")
#             previous_best_model.update(
#                 {k: v for k, v in model_handler_args.items() if v is not None}
#             )
#             model_handler_args = previous_best_model
#         print("------------------------------------------------------------")
#         print("doing the exp", self.name)
#         print("previous_experiment", previous_experiment)
#         if previous_experiment:
#             print("previous_experiment", previous_experiment.best_case)
#         print("model_handler_args", model_handler_args)
#         print("--------------------------------------------------")

#         self.model_handler = ModelHandler(
#             name=self.name,
#             project_manager=self.project_manager,
#             n_features_predict=self.data_manager.n_features_predict,
#             n_features_train=self.data_manager.n_features_train,
#             target_variable=self.target_variable,
#             data_manager_name=self.data_manager.name,
#             train_fn=self.train_fn,
#             **model_handler_args,
#         )
#         self.model_handlers[self.model_handler.name] = self.model_handler

#         self.experiment_list = self.list_experiments(previous_experiment)
#         self.experiments = self.get_experiment_models()
#         self.save()

#     def list_experiments(self, previous_experiment=None):
#         previous_experiment = previous_experiment or {}
#         previous_best_exp = getattr(previous_experiment, "best_exp", {})
#         previous_best_exp = {**previous_best_exp}
#         optimizer = self.optimizer or previous_best_exp.get(
#             "optimizer", "adam"
#         )
#         loss = self.loss or previous_best_exp.get("loss", MeanSquaredError())
#         batch_size = self.batch_size or previous_best_exp.get(
#             "batch_size", 252
#         )
#         weights = self.weights or previous_best_exp.get("weights", False)
#         parameters_to_list = {
#             "optimizer": optimizer,
#             "loss": loss,
#             "batch_size": batch_size,
#             "weights": weights,
#         }
#         return expand_all_alternatives(parameters_to_list)

#     def name_experiments(self):
#         named_exps = {}
#         for exp in self.experiment_list:
#             opt = exp["optimizer"]
#             los = "".join([f[0] for f in exp["loss"].name.split("_")])
#             bs = str(exp["batch_size"])
#             wt = exp["weights"]
#             exp_name = f"{opt}_{los}_B{bs}_{wt}"
#             named_exps[exp_name] = exp
#         return named_exps

#     def get_experiment_models(self):
#         experiments = {}
#         for model_handler in self.model_handlers.values():
#             for model_arch in model_handler.models_confs.keys():
#                 for case, case_args in self.name_experiments().items():
#                     case_name = f"{model_arch}_{case}"
#                     experiments[case_name] = case_args.copy()
#                     experiments[case_name]["model"] = f"{model_arch}"
#                     experiments[model_arch][
#                         "model_handler"
#                     ] = model_handler.name

#         return experiments

#     def train_experiment(self):
#         if not getattr(self, "experiments", False):
#             self.obj_setup()
#         for exp, exp_args in self.experiments.items():
#             exp_fit_args = {**exp_args}
#             if "model" in exp_fit_args:
#                 exp_fit_args.pop("model")
#             if "model_handler" in exp_fit_args:
#                 model_handler_name = exp_fit_args.pop("model_handler")

#             self.model_handlers[model_handler_name].train_model(
#                 exp,
#                 self.epochs,
#                 self.data_manager,
#                 callbacks=self.callbacks,
#                 **exp_fit_args,
#             )

#     def validate_experiment(self, **kwargs):
#         print(kwargs)
#         exp_results_path = os.path.join(
#             self.work_folder, "experiment_score.csv"
#         )
#         if os.path.exists(exp_results_path):
#             exp_results = pd.read_csv(exp_results_path, index_col=0)
#         else:
#             exp_results = pd.DataFrame()
#         for exp in self.experiments.keys():
#             if "name" in exp_results:
#                 saved_exp_score = exp_results[exp_results["name"] == exp]
#                 num_exps = len(saved_exp_score)
#             else:
#                 saved_exp_score = None
#                 num_exps = 0

#             if num_exps < self.epochs:
#                 exp_score = self.model_handler.validate_model(
#                     exp,
#                     self.validation_fn,
#                     self.data_manager,
#                     old_score=saved_exp_score,
#                     **kwargs,
#                 )
#                 exp_results = pd.concat([exp_results, exp_score])
#         exp_results = exp_results.drop_duplicates(["name", "epoch"])
#         exp_results = exp_results.reset_index(drop=True)
#         exp_results.to_csv(exp_results_path)
#         return exp_results

#     def validate_results(
#         self,
#         exp_results=None,
#         result_validation_fn=None,
#         validation_target=None,
#         metrics_to_save_fn=None,
#         **kwargs,
#     ):
#         if exp_results is None:
#             exp_results = self.validate_experiment()
#         result_validation_fn = (
#             result_validation_fn or self.result_validation_fn
#         )
#         validation_target = validation_target or self.validation_target
#         exp_results = exp_results[
#             exp_results["name"].isin(self.experiments.keys())
#         ]
#         exp_results = exp_results[exp_results["epoch"] > 50]
#         exp_count = kwargs.pop("exp_count", None)
#         self.best_case, self.best_result = result_validation_fn(
#             exp_results, validation_target, **kwargs
#         )
#         if len(self.best_case) > 0:
#             exp_results = exp_results[
#                 exp_results["name"].isin(self.best_case.name.unique())
#             ]
#             exp_count = kwargs.pop("exp_count", None)
#             self.best_case, self.best_result = result_validation_fn(
#                 exp_results, "rmse", **kwargs
#             )

#         self.best_exp = self.experiments[self.best_case.name.item()]
#         if self.final_experiment:
#             best_model_path = os.path.join(
#                 self.model_handler.work_folder,
#                 self.best_case.name.item(),
#                 "freq_saves",
#                 f"{self.best_case.epoch.item()}.keras",
#             )
#             final_model_path = os.path.join(
#                 self.project_manager.final_models_folder,
#                 self.target_variable,
#                 self.data_manager.name,
#                 "final_model.keras",
#             )
#             os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
#             print(
#                 "ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo"
#             )
#             print(best_model_path)
#             shutil.copy(best_model_path, final_model_path)
#         if metrics_to_save_fn:
#             self.exp_metrics = metrics_to_save_fn(
#                 experiment=self, exp_results=exp_results
#             )
#         if exp_count is not None:
#             if self.target_variable not in self.project_manager.history:
#                 self.project_manager.history[self.target_variable] = {}
#             if (
#                 self.data_manager.name
#                 not in self.project_manager.history[self.target_variable]
#             ):
#                 self.project_manager.history[self.target_variable][
#                     self.data_manager.name
#                 ] = {}
#             self.project_manager.history[self.target_variable][
#                 self.data_manager.name
#             ][f"{exp_count}"] = {
#                 "best_model": self.best_case.name.item(),
#                 "epoch": self.best_case.epoch.item(),
#                 "best_value": self.best_case[validation_target].item(),
#             }
#             self.project_manager.save()
#         self.save()
#         return self.best_case

#     def write_report(self, exp_results=None, benchmark_data=None, **kwargs):
#         if exp_results is None:
#             exp_results = self.validate_experiment()
#         benchmark_score = None
#         if benchmark_data is None:
#             benchmark_data = os.path.join(
#                 self.data_manager.work_folder,
#                 "benchmark",
#                 self.target_variable,
#                 "benchmark.json",
#             )
#             benchmark_score = load_json_dict(benchmark_data)
#             benchmark_data = np.load(benchmark_data.replace("json", "npz"))

#         folder_figures = kwargs.pop(
#             "folder_figures",
#             self.work_folder.replace("/experiment/", "/reports/"),
#         )
#         figure_name = kwargs.pop(
#             "figure_name", f"experiment_results_{self.target_variable}.png"
#         )
#         limit_by = kwargs.pop("limit_by", "benchmark")
#         metrics_to_check = kwargs.pop("metrics_to_check", None)
#         os.makedirs(folder_figures, exist_ok=True)
#         exp_results = exp_results[
#             exp_results["name"].isin(self.experiments.keys())
#         ]
#         self.write_report_fn(
#             exp_results,
#             metrics_to_check=metrics_to_check,
#             benchmark_score=benchmark_score,
#             folder_figures=folder_figures,
#             figure_name=figure_name,
#             limit_by=limit_by,
#             **kwargs,
#         )
#         # TODO: write validation vs best model (year, month, day : worst and best)


# class StatsExperiment(BaseExperiment):
#     pass


# class ExperimentHandler(ShaiHulud):
#     registry = dict()

#     def __init__(self):
#         pass

#     def __new__(cls, model_backend: str, **kwargs):
#         # Dynamically create an instance of the specified model class
#         model_backend = model_backend.lower()
#         exphandler_class = ExperimentHandler.registry[model_backend]
#         instance = super().__new__(exphandler_class)
#         # Inspect the __init__ method of the model class to get its parameters
#         init_params = inspect.signature(cls.__init__).parameters
#         # Separate kwargs based on the parameters expected by the model's __init__
#         modelhandler_kwargs = {
#             k: v for k, v in kwargs.items() if k in init_params
#         }
#         modelbackend_kwargs = {
#             k: v for k, v in kwargs.items() if k not in init_params
#         }

#         for name, method in cls.__dict__.items():
#             if "__" in name:
#                 continue
#             if callable(method) and hasattr(instance, name):
#                 instance.__dict__[name] = method.__get__(instance, cls)

#         cls.__init__(instance, **modelhandler_kwargs)
#         instance.__init__(**modelbackend_kwargs)

#         return instance

#     @staticmethod
#     def register(constructor):
#         # TODO: only register if its a BaseModel subclass
#         ExperimentHandler.registry[
#             constructor.__name__.lower().replace("handler", "")
#         ] = constructor


# ExperimentHandler.register(KerasExperiment)
# ExperimentHandler.register(StatsExperiment)


def ExperimentFactory(
    data_handlers=None,
    target_variables=None,
    previous_experiment_dict=None,
    **kwargs,
):
    previous_experiment_dict = previous_experiment_dict or {}
    experiment_dict = {}
    # Make it so that it allows for diff variables on multiplr targ exp
    for target_project in data_handlers.keys():
        data_manager = data_handlers[target_project]
        target_variable = data_manager.target_variable
        previous_experiment = previous_experiment_dict.get(
            target_project, None
        )
        experiment_handles = ExperimentHandler(
            target_variable=target_variable,
            data_manager=data_manager,
            previous_experiment=previous_experiment,
            **kwargs,
        )

        experiment_dict[target_project] = experiment_handles

    return experiment_dict
