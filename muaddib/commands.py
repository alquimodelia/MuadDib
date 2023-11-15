import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from cookiecutter.main import cookiecutter
from cookiecutter.repository import determine_repo_dir

from muaddib.utils import startup


def open_new_console(command):
    if sys.platform == "win32":
        process = subprocess.Popen(f"start cmd /K {command}", shell=True)
    elif sys.platform == "darwin":
        process = subprocess.Popen(
            f'osascript -e \'tell app "Terminal" to do script "{command}"\'',
            shell=True,
        )
    else:
        try:
            process = subprocess.Popen(f"{command}", shell=True)
        except FileNotFoundError:
            try:
                process = subprocess.Popen(f"xterm -e {command}", shell=True)
            except FileNotFoundError:
                print("Unsupported platform")
                sys.exit(1)
    process.wait()  # Wait for the process to finish


def new(args):
    TEMPLATE_TO_BUILD = (
        "https://github.com/alquimodelia/arrakis-coockiecutter.git"
    )
    directory = (
        args.template_name
    )  # specify the directory inside the repository
    checkout = (
        None  # specify the tag, branch, or commit to checkout, if necessary
    )

    with TemporaryDirectory() as tmpdir:
        try:
            cookiecutter_dir, _ = determine_repo_dir(
                template=TEMPLATE_TO_BUILD,
                abbreviations={},
                clone_to_dir=Path(tmpdir).resolve(),
                checkout=checkout,
                no_input=True,
                directory=directory,
            )
        except Exception as exc:
            raise Exception(
                f"Failed to generate project: could not clone repository at {TEMPLATE_TO_BUILD}."
            ) from exc

        project_path = cookiecutter(
            template=str(cookiecutter_dir),
            no_input=True,
            extra_context={"project_name": args.project_name},
        )
        print(f"Creating new project: {args.project_name}")
        return project_path


def start(args):
    # TODO: Check if its in a muaddib project

    startup()
    global experiments_list
    global experiments_dict
    from experiments.experiment import experiments_dict, experiments_list

    print("uipiiiiiii", experiments_dict)


def init(args):
    start(args)
    subprocess.run(["pip", "install", "-e", "."])


def process_data(args):
    pass


def process_models(args):
    pass


# TODO: Create reporting command, using the validations


def train(args, train_model=None):
    # TODO: REFACTOR!
    if train_model is None:
        print("normal triaing")
        train_script_path = os.path.join(MODEL_SCRIPTS_DIR, "train_model.py")
        if not os.path.exists(train_script_path):
            print(f"Training script not found: {train_script_path}")
            return
        from train_model import train_model

    params = {
        k: v
        for k, v in vars(args).items()
        if k not in ["logconf", "command", "case"]
    }
    train_model(**params)


def train_case(args):
    start(args)
    # TODO: handle this multiple imports

    experiment_name = args.experiment
    case_name = args.case
    exp = experiments_dict[experiment_name]
    exp.setup()
    case_obj = exp.study_cases[case_name]
    if not case_obj.complete:
        print("-------------------------------------------------------------")
        print(f"Training Model:{case_name}. On experiment {experiment_name}")
        case_obj.train_model()


# BaseManager.register('Experiment', Experiment)
# BaseManager.register('Case', Case)


def train_model_process(case_obj):
    if not case_obj.complete:
        print("-------------------------------------------------------------")
        print(
            f"Training Model:{case_obj.name}. On experiment {case_obj.experiment_name}"
        )
        case_obj.train_model()


# TODO: chane this, this is garbage
def train_on_experiment_loop(args):
    start(args)
    experiment_name_train = args.experiment
    case_name_train = args.case
    for experiment_name, exp in experiments_dict.items():
        exp.setup()
        print("-------------------------------------------------------------")
        for case_obj in exp.conf:
            if experiment_name == experiment_name_train:
                case_name = str(case_obj.name)
                if len(case_name.split("_")) > 4:
                    continue
                # if "252" in case_obj.name:
                #     continue
                if case_obj.name.endswith("_adam"):
                    continue

                if case_name == case_name_train:
                    if not case_obj.complete:
                        case_obj.train_model()
                        return
                        # command = (
                        #     f"KERAS_BACKEND={case_obj.keras_backend}"
                        #     f" muaddib train_case --experiment={experiment_name} --case={case_name}"
                        # )

                        # open_new_console(command)
            case_obj.validate_model()

        exp.validate_experiment()
        exp.visualize_report()


def experiment(args):
    start(args)
    name = getattr(args, "name", None)

    # manager = BaseManager()
    # manager.start()

    for experiment_name, exp in experiments_dict.items():
        # Create a proxy for the ExperimentClass instance
        # exp = manager.Experiment(exp)
        exp.setup()
        print("-------------------------------------------------------------")
        # if exp.complete:
        # continue
        if name is not None:
            if name != experiment_name:
                continue
        print(exp.name)
        for case_obj in exp.conf:
            print(case_obj.name)

            case_name = str(case_obj.name)

            # case_obj = manager.Case(exp)
            # if not case_obj.complete:
            #     with Pool(processes=1) as pool:
            #         argst = Namespace(experiment=experiment_name, case=case_name)
            #         # pool.apply(train_model_process, args=(case_obj,))
            #         pool.apply(train_case, args=(argst,))

            # case_obj.validate_model()

            if not case_obj.complete:
                # command = (
                #     f"KERAS_BACKEND={case_obj.keras_backend}"
                #     f" muaddib train_case --experiment={experiment_name} --case={case_name}"
                # )
                command = (
                    f"KERAS_BACKEND={case_obj.keras_backend}"
                    f" muaddib train_on_experiment_loop --experiment={experiment_name} --case={case_name}"
                )
                open_new_console(command)
            case_obj.validate_model()
        exp.validate_experiment()
        exp.visualize_report()
        print("exp.worthy_cases in commandsssssssssssssssss")

        print(exp.worthy_cases)
