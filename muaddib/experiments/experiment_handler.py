import os

import numpy as np
import pandas as pd

from muaddib.experiments.default_functions import (
    make_experiment_plot,
    make_tex_table_best_result,
)
from muaddib.models.model_handler import ModelHandler
from muaddib.muaddib import ShaiHulud
from muaddib.shaihulud_utils import AdvanceLossHandler


class ExperimentHandler(ShaiHulud):
    listing_conf_properties = ["model_handlers", "experiments"]
    single_conf_properties = [
        "project_manager",
        "target_variable",
        "data_manager",
        "callbacks",
        "metric_scores_fn",
        "validation_target",
        "result_validation_fn",
        "write_report_fn",
    ]
    columns_model_args = [
        "arch",
        "activation_middle",
        "activation_end",
        "x_timesteps",
        "y_timesteps",
        "filters",
        "features",
        "classes",
        "optimizer",
        "loss",
        "batch",
        "weights",
        "p",
        "P",
        "q",
        "Q",
        "d",
        "D",
        "s",
        "trend",
    ]

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
        use_suggestions=True,
        exp_col=None,
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
            self.validation_target = validation_target
            self.previous_experiment = previous_experiment
            self.write_report_fn = write_report_fn or make_experiment_plot
            self.final_experiment = final_experiment
            self.args_in_exp = []
            self.use_suggestions = use_suggestions
            self.exp_col = exp_col
            if not isinstance(self.exp_col, list):
                self.exp_col = [self.exp_col]

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
            # #     "x_timesteps": data_manager.x_timesteps,
            # #     "y_timesteps": data_manager.y_timesteps + extra_y_timesteps,
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
            for kwarg, argument in kwargs.items():
                if self.use_suggestions is not False:
                    suggetion_name = f"suggested_{kwarg}"
                    suggestion_arg = getattr(
                        self.data_manager, suggetion_name, None
                    )
                    if suggestion_arg is not None:
                        # TODO: make a forcing list function
                        if not isinstance(suggestion_arg, list):
                            suggestion_arg = list(set([suggestion_arg]))
                            sug_size = len(suggestion_arg)
                            if isinstance(self.use_suggestions, int):
                                sug_size = self.use_suggestions
                            suggestion_arg = suggestion_arg[:sug_size]
                        if not isinstance(argument, list):
                            argument = [argument]
                        argument = argument + suggestion_arg
                        argument = list(set(argument))
                arg_to_add = argument
                if (
                    kwarg in model_handler.model_args
                    or kwarg in model_handler.fit_kwargs
                    or kwarg in model_handler.class_args
                ):
                    if kwarg in model_handler.model_args:
                        self.args_in_exp.append(kwarg)
                    if kwarg in original_kwargs:
                        original_kwargs.pop(kwarg)

                    # Check if the argment is already there an add if not.
                    if kwarg in model_handler_kwargs:
                        if argument == model_handler_kwargs[kwarg]:
                            continue
                        old_arg = model_handler_kwargs[kwarg]
                        if not isinstance(old_arg, list):
                            old_arg = [old_arg]
                        new_arg = argument
                        if not isinstance(new_arg, list):
                            new_arg = [new_arg]
                        for n_arg in new_arg:
                            if n_arg not in old_arg:
                                old_arg.append(n_arg)
                        arg_to_add = old_arg
                    model_handler_kwargs[kwarg] = arg_to_add

        self.model_handler_kwargs = model_handler_kwargs
        return original_kwargs

    # def get_model_handler_kwargs(self, kwargs):
    #     original_kwargs = {**kwargs}
    #     model_handler_kwargs = {}
    #     for model_handler_name, model_handler in ModelHandler.registry.items():
    #         if self.previous_experiment:
    #             previous_experiment_args = {}
    #             best_case = self.previous_experiment.best_case
    #             if model_handler_name==best_case["model_handler_name"]:
    #                 model_handler_kwargs.update(best_case["model_conf"])
    #                 model_handler_kwargs.update(best_case["exp_conf"])

    #         for kwarg, argument in kwargs.items():
    #             arg_to_add = argument
    #             if not isinstance(arg_to_add, list):
    #                 arg_to_add=[arg_to_add]
    #             # Only add archs from the modelhandler
    #             if kwarg=="archs":
    #                 usable_archs = []
    #                 for arch in arg_to_add:
    #                     if arch.lower() in [f.lower() for f in model_handler.model_archs]:
    #                         usable_archs.append(arch)
    #                 if len(usable_archs)==0:
    #                     continue
    #                 arg_to_add=usable_archs
    #             # check if it is already from previous
    #             if kwarg in model_handler_kwargs:
    #                 old_arg = model_handler_kwargs[kwarg]
    #                 if not isinstance(old_arg, list):
    #                     old_arg=[old_arg]
    #                 for old in old_arg:
    #                     if old not in arg_to_add:
    #                         arg_to_add.append(old)
    #             # Now add if its an arg from the current handler
    #             if (
    #                 kwarg in model_handler.model_args
    #                 or kwarg in model_handler.fit_kwargs
    #                 or kwarg in model_handler.class_args
    #             ):
    #                 if kwarg in original_kwargs:
    #                     original_kwargs.pop(kwarg)
    #                 model_handler_kwargs[kwarg] = arg_to_add
    #     experiment_model_handler = ModelHandler.create_model_handlers(
    #         **model_handler_kwargs,
    #         project_manager=self.project_manager,
    #         datamanager=self.data_manager,
    #     )
    #     self.model_handlers = experiment_model_handler
    #     return original_kwargs

    def obj_setup(self, model_handler_args=None, previous_experiment=None):
        model_handler_kwargs = {}
        for model_handler_name, model_handler in ModelHandler.registry.items():
            if self.previous_experiment:
                best_case = self.previous_experiment.best_case
                if model_handler_name == best_case["model_handler_name"]:
                    model_handler_kwargs.update(best_case["model_conf"])
                    model_handler_kwargs.update(best_case["exp_conf"])

                for kwarg, argument in self.model_handler_kwargs.items():
                    arg_to_add = argument
                    if (
                        kwarg in model_handler.model_args
                        or kwarg in model_handler.fit_kwargs
                        or kwarg in model_handler.class_args
                    ):

                        # Check if the argment is already there an add if not.
                        if kwarg in model_handler_kwargs:
                            if argument == model_handler_kwargs[kwarg]:
                                continue
                            old_arg = model_handler_kwargs[kwarg]
                            if not isinstance(old_arg, list):
                                old_arg = [old_arg]
                            new_arg = argument
                            if isinstance(new_arg, AdvanceLossHandler):
                                new_arg.set_previous_loss(
                                    self.previous_experiment.best_case[
                                        "exp_conf"
                                    ]["loss"]
                                )
                                new_arg = [f for f in new_arg]
                            if not isinstance(new_arg, list):
                                new_arg = [new_arg]
                            for n_arg in new_arg:
                                if n_arg not in old_arg:
                                    # Avoid loss duplication, BUG: but this migh happen with other functions
                                    if kwarg == "loss":
                                        if n_arg.name in [
                                            f.name for f in old_arg
                                        ]:
                                            continue
                                    old_arg.append(n_arg)
                            arg_to_add = old_arg
                        model_handler_kwargs[kwarg] = arg_to_add
            else:
                model_handler_kwargs = self.model_handler_kwargs

        experiment_model_handler = ModelHandler.create_model_handlers(
            name=self.name,
            **model_handler_kwargs,
            project_manager=self.project_manager,
            datamanager=self.data_manager,
        )
        self.model_handlers = experiment_model_handler

        experiments = {}
        for key, val in self.model_handlers.items():
            for case_name in val.exp_cases.keys():
                experiments[case_name] = {
                    "model_handler_name": key,
                }
        self.experiments = experiments

    def train_experiment(self):
        if not getattr(self, "experiments", False):
            self.obj_setup()
        print("----------------------------------------------------")
        print(f"Training {self.name}")
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
            print(f"Training {exp}")
            self.model_handlers[model_handler_name].train_model(
                exp,
                datamanager=self.data_manager,
                **exp_fit_args,
            )
            print("****************************************************")
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    def validate_experiment(self, **kwargs):
        exp_results_path = os.path.join(
            self.work_folder, "experiment_score.csv"
        )
        if os.path.exists(exp_results_path):
            exp_results = pd.read_csv(exp_results_path)
        else:
            exp_results = pd.DataFrame()
        for exp in self.experiments.keys():
            # Check for model inference results and does the inference if not done
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

    def create_experiment_validation_data(self, exp_results, biggest=True):
        exp_prediction_validation_path = os.path.join(
            self.work_folder,
            f"prediction_validation_{self.validation_target}.npz",
        )
        if os.path.exists(exp_prediction_validation_path):
            exp_prediction_validation = dict(
                np.load(exp_prediction_validation_path)
            )
        else:
            exp_prediction_validation = {}
        for exp in self.experiments.keys():
            # Check for model inference results and does the inference if not done
            if exp in exp_prediction_validation:
                continue
            else:
                model_handler_name = self.experiments[exp][
                    "model_handler_name"
                ]
                case_result = exp_results[exp_results["name"] == exp]
                if biggest:
                    best_ind = (
                        case_result[self.validation_target].nlargest(1).index
                    )
                else:
                    best_ind = (
                        case_result[self.validation_target].nsmallest(1).index
                    )
                best_epoch = int(case_result.loc[best_ind].epoch.item())

                # Check the npz file created for model case and add to experiment npz
                exp_npz_file_path = os.path.join(
                    self.model_handlers[model_handler_name].work_folder,
                    exp,
                    "predictions.npz",
                )
                exp_case_validation = dict(np.load(exp_npz_file_path))
                for key, arr in exp_case_validation.items():
                    if "prediction" not in key:
                        if key not in exp_prediction_validation:
                            exp_prediction_validation[key] = arr
                    else:
                        epoch = int(key.replace("prediction_", ""))
                        if best_epoch == epoch:
                            exp_prediction_validation[exp] = arr

        np.savez_compressed(
            exp_prediction_validation_path, **exp_prediction_validation
        )

    def validate_results(
        self, exp_results, validation_target=None, biggest=True
    ):
        validation_target = validation_target or self.validation_target
        exp_results = exp_results[
            exp_results["name"].isin(self.experiments.keys())
        ]
        if biggest:
            best_ind = exp_results[validation_target].nlargest(1).index
        else:
            best_ind = exp_results[validation_target].nsmallest(1).index
        best_name = exp_results.loc[best_ind].name.item()
        best_epoch = exp_results.loc[best_ind].epoch.item()
        model_handler_name = self.experiments[best_name]["model_handler_name"]
        # best_name=self.best_case
        for mod_conf in self.model_handlers[
            model_handler_name
        ].models_confs.keys():
            # The model_conf_name is part of the best_name, as it is made my model name and fit args name
            if mod_conf in best_name:
                model_conf_name = mod_conf
                model_conf = self.model_handlers[
                    model_handler_name
                ].models_confs[model_conf_name]
        self.best_case = {
            "name": best_name,
            "model_handler_name": model_handler_name,
            "model_conf": model_conf,
            "exp_conf": self.model_handlers[model_handler_name].exp_cases[
                best_name
            ],
            "epoch": best_epoch,
        }
        self.create_experiment_validation_data(exp_results, biggest=biggest)
        self.save()

    def write_report(self, exp_results=None, **kwargs):
        if exp_results is None:
            exp_results = self.validate_experiment()

        folder_figures = kwargs.pop(
            "folder_figures",
            self.work_folder.replace("/experiment/", "/reports/"),
        )

        # limit_by = kwargs.pop("limit_by", "benchmark")
        # metrics_to_check = kwargs.pop("metrics_to_check", None)
        os.makedirs(folder_figures, exist_ok=True)
        # exp_results = exp_results[
        #     exp_results["name"].isin(self.experiments.keys())
        # ]
        metrics_to_keep = [
            f
            for f in exp_results.columns
            if f not in [*self.columns_model_args, "name"]
        ]
        path_to_save = os.path.join(folder_figures, "exp_result.tex")
        make_tex_table_best_result(
            exp_results,
            path_to_save,
            exp_col=self.exp_col,
            metric=self.validation_target,
            metrics_to_keep=metrics_to_keep,
        )
        exp_results = exp_results[
            [
                *[
                    f
                    for f in exp_results.columns
                    if f not in self.columns_model_args
                ]
                + self.exp_col,
            ]
        ]
        self.write_report_fn(
            exp_results,
            folder_figures=folder_figures,
            column_to_group=self.exp_col,
            **kwargs,
        )

    def run_experiment(
        self,
        train_args=None,
        validate_experiment_args=None,
        validate_results_args=None,
        write_report_args=None,
    ):
        train_args = train_args or {}
        validate_experiment_args = validate_experiment_args or {}
        validate_results_args = validate_results_args or {}
        write_report_args = write_report_args or {}

        self.train_experiment(**train_args)
        results = self.validate_experiment(**validate_experiment_args)
        self.validate_results(results, **validate_results_args)
        self.write_report(results, **write_report_args)


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
#                 "x_timesteps": data_manager.x_timesteps,
#                 "y_timesteps": data_manager.y_timesteps + extra_y_timesteps,
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
#             previous_best_model.pop("num_classes")
#             previous_best_model.pop("num_features_to_train")
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
#             num_classes=self.data_manager.num_classes,
#             num_features_to_train=self.data_manager.num_features_to_train,
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


# def ExperimentFactory(
#     data_handlers=None,
#     target_variables=None,
#     previous_experiment_dict=None,
#     **kwargs,
# ):
#     previous_experiment_dict = previous_experiment_dict or {}
#     experiment_dict = {}
#     # Make it so that it allows for diff variables on multiplr targ exp
#     for target_project in data_handlers.keys():
#         data_manager = data_handlers[target_project]
#         target_variable = data_manager.target_variable
#         previous_experiment = previous_experiment_dict.get(
#             target_project, None
#         )
#         experiment_handles = ExperimentHandler(
#             target_variable=target_variable,
#             data_manager=data_manager,
#             previous_experiment=previous_experiment,
#             **kwargs,
#         )

#         experiment_dict[target_project] = experiment_handles

#     return experiment_dict


class ExperimentFactory:
    def __init__(
        self,
        data_handlers=None,
        target_variables=None,
        previous_experiment_dict=None,
        new_obj=True,
        **kwargs,
    ):
        previous_experiment_dict = previous_experiment_dict or {}
        experiment_dict = {}
        if new_obj:
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
        self.data_handlers = data_handlers
        self.target_variables = target_variables
        self.previous_experiment_dict = previous_experiment_dict
        self.experiment_dict = experiment_dict

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(
                "Can only add another ExperimentFactory type objects."
            )

        combined_targets = [f for f in self.experiment_dict.keys()] + [
            f for f in other.experiment_dict.keys()
        ]
        combined_targets = list(set(combined_targets))
        experiment_dict = {}
        for target_project in combined_targets:
            self_experiment = self.experiment_dict.get(target_project, None)
            other_experiment = other.experiment_dict.get(target_project, None)
            if self_experiment is None:
                if other_experiment is not None:
                    combined_experiment = other_experiment
            elif other_experiment is None:
                combined_experiment = self_experiment
            else:
                combined_experiment = self_experiment + other_experiment
            experiment_dict[target_project] = combined_experiment
        combined_obj = ExperimentFactory(
            new_obj=False,
        )
        combined_obj.experiment_dict = experiment_dict

        return combined_obj
