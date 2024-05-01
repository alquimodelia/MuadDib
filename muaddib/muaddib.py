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
