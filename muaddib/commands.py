import os
import subprocess
import sys

from cookiecutter.main import cookiecutter

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
    # TODO: redo to use cookicutter from arrakis-cookicutter
    cookiecutter_file = os.path.join(PROJECT_COOKICUTTER, "cookiecutter.json")

    if not os.path.exists(PROJECT_COOKICUTTER):
        print(f"Template directory not found: {PROJECT_COOKICUTTER}")
        return

    if not os.path.exists(cookiecutter_file):
        print(f"cookiecutter.json not found in: {PROJECT_COOKICUTTER}")
        return

    project_path = cookiecutter(
        template=PROJECT_COOKICUTTER,
        no_input=True,
        extra_context={"project_name": args.project_name},
    )
    print(f"Creating new project: {args.project_name}")


def start(args):
    # TODO: Check if its in a muaddib project

    startup()


def init(args):
    start(args)
    subprocess.run(["pip", "install", "-e", "."])


def process_data(args):
    pass


def process_models(args):
    pass


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

    from experiments.experiment import experiments_list

    experiment_name = args.experiment
    case_name = args.case
    # TODO: add a dict thingy yo this
    exp = [f for f in experiments_list if f.name == experiment_name][0]
    case_obj = [f for f in exp.conf if f.name == case_name][0]
    if not case_obj.complete:
        print("-------------------------------------------------------------")
        print(f"Training Model:{case_name}. On experiment {experiment_name}")
        case_obj.train_model()


def experiment(args):
    start(args)
    name = getattr(args, "name", None)

    from experiments.experiment import experiments_list

    for exp in experiments_list:
        print("-------------------------------------------------------------")
        experiment_name = exp.name
        print(f"Experiment {experiment_name}")
        # if exp.complete:
        # continue
        if name is not None:
            if name != experiment_name:
                continue
        for case_obj in exp.conf:
            if not case_obj.complete:
                case_name = str(case_obj.name)

                command = (
                    f"KERAS_BACKEND={case_obj.keras_backend}"
                    f" muaddib train_case --experiment={experiment_name} --case={case_name}"
                )

                open_new_console(command)
            case_obj.validate_model()

        exp.validate_experiment()
