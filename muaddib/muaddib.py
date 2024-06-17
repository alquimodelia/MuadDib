import os
import pathlib

import tinydb

from muaddib.tinydb_serializers import create_class_serializer, serialization


class ShaiHulud:
    """
    A class that represents a model.

    Attributes
    ----------
    p1 : str
        Description of the attribute

    Methods
    -------
    my_method(p2)
        Description of the method
    """

    def __init__(
        self,
        work_folder=None,
        **kwargs,
    ):
        """
        Constructor method.

        Parameters
        ----------
        p1 : str, optional
            Description of the parameter, by default "whatever"
        """
        # TODO: refactor whole read/write pipeline
        self.work_folder = work_folder or str(pathlib.Path("").resolve())
        conf_file = getattr(self, "conf_file", None)
        self.conf_file = conf_file
        if conf_file and os.path.exists(conf_file):
            self.load(conf_file)
        else:
            self.setup(**kwargs)

    def get_vars_to_save(self):
        return None

    def save(self):
        serialization.register_serializer(
            create_class_serializer(ShaiHulud), "ShaiHulud"
        )
        db = tinydb.TinyDB(
            self.conf_file, storage=serialization, indent=4, sort_keys=True
        )
        vars_to_save = self.get_vars_to_save() or vars(self)
        records = db.all()
        if records:
            record = records[0]  # assuming you only have one record
            record.update(vars_to_save)
            db.update(record)
        else:
            db.insert(vars_to_save)

    def setup_after_load(self):
        pass

    def load(self, conf_file):
        serialization.register_serializer(
            create_class_serializer(ShaiHulud), "ShaiHulud"
        )
        db = tinydb.TinyDB(
            conf_file, storage=serialization, indent=4, sort_keys=True
        )
        record = db.all()[0]  # assuming you only have one record
        for attr, value in record.items():
            setattr(self, attr, value)
        self.setup_after_load()

    def obj_setup(self, **kwargs):
        pass

    def setup(self, **kwargs):
        obj_setup_args = kwargs.pop("obj_setup_args", {})
        if not self.conf_file:
            filename = self.name or self.__class__.__name__
            self.conf_file = os.path.join(
                self.work_folder, f"{filename}_conf.json"
            )
            os.makedirs(os.path.dirname(self.conf_file), exist_ok=True)
        for kwarg, value in kwargs.items():
            setattr(self, str(kwarg), value)
        # Only setup the experiment if there is
        if not hasattr(self, "previous_experiment"):
            self.obj_setup(**obj_setup_args)
        # BUG: mirrow weights serialeries not working with the inner loss functions
        self.save()

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError("Can only add same ShaiHulud type objects.")

        combined_config = {}
        combined_shai_name = f"{self.name}_{other.name}"
        combined_config["name"] = combined_shai_name

        for single_conf in self.single_conf_properties:
            single_attr = getattr(
                self, single_conf, getattr(other, single_conf, None)
            )
            if single_attr is not None:
                combined_config[single_conf] = single_attr
        # Create a new instance of ShaiHulud to hold the combined configuration
        combined_shai = self.__class__(**combined_config)

        combined_attrs = getattr(self, "listing_conf_properties", [])
        for attr in combined_attrs:
            # TODO: add model handlers!!
            # TODO: recursive dict vs list
            self_attr = getattr(self, attr, [])
            other_attr = getattr(other, attr, [])
            if not isinstance(self_attr, dict):
                combined_attr = self_attr + other_attr
            else:
                combined_attr = {}
                combined_targets = list(
                    set(
                        [f for f in self_attr.keys()]
                        + [f for f in other_attr.keys()]
                    )
                )

                for key in combined_targets:
                    if key not in other_attr:
                        combined_attr[key] = self_attr[key]
                    elif key not in self_attr:
                        combined_attr[key] = other_attr[key]
                    else:
                        if not isinstance(
                            self_attr[key], dict
                        ) and not isinstance(other_attr[key], dict):
                            combined_attr[key] = (
                                self_attr[key] + other_attr[key]
                            )
                        elif isinstance(other_attr[key], dict):
                            combined_attr[key] = other_attr[key]

                        # else:
                        #     comb_res = {}
                        #     combined_second_targets = list(
                        #             set(
                        #                 [f for f in self_attr[key].keys()]
                        #                 + [f for f in other_attr[key].keys()]
                        #             )
                        #         )
                        #     for sec_targ in combined_second_targets:
                        #         if sec_targ not in other_attr[key]:
                        #             comb_res[sec_targ] = self_attr[key][sec_targ]
                        #         elif sec_targ not in self_attr[key]:
                        #             comb_res[sec_targ] = other_attr[key][sec_targ]
                        #         else:
                        #             self_sec_attr = self_attr[key][sec_targ]
                        #             other_sec_attr = other_attr[key][sec_targ]
                        #             if not isinstance(self_sec_attr, list):

                        #             comb_res[sec_targ] = ,

            setattr(combined_shai, attr, combined_attr)
        # # Sum up the configurations
        # combined_config = {**self.get_vars_to_save(), **other.get_vars_to_save()}
        # combined_shai.setup(**combined_config)

        return combined_shai


class ProjectFolder(ShaiHulud):
    def __init__(
        self,
        name=None,
        root_folder=None,
        target_variables=None,
        **kwargs,
    ):
        self.name = name or "muaddib_project"
        root_folder = root_folder or "."
        self.root_folder = pathlib.Path(root_folder).absolute()
        self.target_variables = target_variables

        self.data_folder = self.root_folder.joinpath("data")
        self.experiment_folder = self.root_folder.joinpath("experiment")
        self.models_folder = self.root_folder.joinpath("models")
        self.notebooks_folder = self.root_folder.joinpath("notebooks")
        self.references_folder = self.root_folder.joinpath("references")
        self.reports_folder = self.root_folder.joinpath("reports")
        self.project_folder = self.root_folder.joinpath(self.name)

        self.model_configuration_folder = self.models_folder.joinpath(
            "configurations"
        )
        self.trained_models_folder = self.models_folder.joinpath(
            "trained_models"
        )
        self.final_models_folder = self.models_folder.joinpath("final_models")

        self.trained_models_folder_variables = []
        self.final_models_folder_variables = []
        self.reports_folder_variables = []
        self.experiments_variables_variables = []

        self.history = {}

        for target_variable in self.target_variables:
            self.trained_models_folder_variables.append(
                self.trained_models_folder.joinpath(target_variable)
            )
            self.final_models_folder_variables.append(
                self.final_models_folder.joinpath(target_variable)
            )
            self.reports_folder_variables.append(
                self.reports_folder.joinpath(target_variable)
            )
            self.experiments_variables_variables.append(
                self.experiment_folder.joinpath(target_variable)
            )

        super().__init__(work_folder=self.root_folder, **kwargs)
        self.check_and_mkdirs()

    def check_and_mkdirs(self):
        for name, var in vars(self).items():
            if isinstance(var, pathlib.Path):
                os.makedirs(var, exist_ok=True)
                setattr(self, name, str(var))
        for folder in (
            self.trained_models_folder_variables
            + self.reports_folder_variables
            + self.experiments_variables_variables
            + self.final_models_folder_variables
        ):
            if isinstance(folder, pathlib.Path):
                os.makedirs(folder, exist_ok=True)
