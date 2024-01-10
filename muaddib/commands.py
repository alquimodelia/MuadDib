import glob
import os
import subprocess
import sys
from tempfile import TemporaryDirectory

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
    TEMPLATE_TO_BUILD = (
        "https://github.com/alquimodelia/arrakis-coockiecutter.git"
    )
    # For local testing, remove when updating
    TEMPLATE_TO_BUILD = "/home/joao/Documentos/repos/arrakis-coockiecutter/"

    directory = (
        args.template_name
    )  # specify the directory inside the repository
    checkout = (
        None  # specify the tag, branch, or commit to checkout, if necessary
    )
    cookiecutter_dir = os.path.join(TEMPLATE_TO_BUILD, directory)
    with TemporaryDirectory() as tmpdir:
        # try:
        #     cookiecutter_dir, _ = determine_repo_dir(
        #         template=TEMPLATE_TO_BUILD,
        #         abbreviations={},
        #         clone_to_dir=Path(tmpdir).resolve(),
        #         checkout=checkout,
        #         no_input=False,
        #         directory=directory,
        #     )
        # except Exception as exc:
        #     print(f"Failed to generate project: could not clone repository at {TEMPLATE_TO_BUILD}.")
        #     # raise Exception(
        #     #     f"Failed to generate project: could not clone repository at {TEMPLATE_TO_BUILD}."
        #     # ) from exc
        print("fds", cookiecutter_dir)
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


def init(args):
    startup()
    subprocess.run(["pip", "install", "-e", "."])


def reset_configurations(args):
    startup()

    EXPERIMENT_FOLDER = os.getenv("EXPERIMENT_FOLDER", None)
    qry = f"{EXPERIMENT_FOLDER}/**/**_conf.json"
    # qry = f"{EXPERIMENT_FOLDER}/**/**experiment_conf.json"
    list_of_all_conf_file = glob.glob(qry, recursive=True)
    # list_of_all_conf_file = [f for f in list_of_all_conf_file if "case_conf" not in f]
    for cf in list_of_all_conf_file:
        os.remove(cf)
    qry = f"{EXPERIMENT_FOLDER}/**/**predict_score.json"
    list_of_all_conf_file = glob.glob(qry, recursive=True)
    for cf in list_of_all_conf_file:
        os.remove(cf)


def process_data(args):
    startup()
    global ALL_DATA_MANAGERS
    from data.definitions import ALL_DATA_MANAGERS

    for dataman in ALL_DATA_MANAGERS.values():
        dataman.process_data()


def process_benchmark(args):
    startup()
    global ALL_DATA_MANAGERS
    from data.definitions import ALL_DATA_MANAGERS

    for dataman in ALL_DATA_MANAGERS.values():
        dataman.process_benchmark()


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


def train_model_process(case_obj):
    if not case_obj.complete:
        print("-------------------------------------------------------------")
        print(
            f"Training Model:{case_obj.name}. On experiment {case_obj.experiment_name}"
        )
        case_obj.train_model()


def train_on_call(args):
    start(args)
    validation_target = getattr(args, "validation_target", None)
    if validation_target is not None:
        os.environ["VALIDATION_TARGET"] = validation_target

    experiment_name_train = args.experiment
    case_name_train = args.case
    print("-------------------")
    print("Runnig train on call")
    print(experiment_name_train)
    print(case_name_train)
    print("-------------------")

    exp_obj = experiments_dict[experiment_name_train]
    exp_obj.setup()
    case_obj = exp_obj.study_cases[case_name_train]
    run_training = not case_obj.complete or exp_obj.final_experiment
    extra_args = {}
    if run_training:
        print(case_obj.name)
        print(case_obj.epochs)
        print(exp_obj.epochs)

        if case_obj.epochs < exp_obj.epochs:
            case_obj.epochs = exp_obj.epochs
            extra_args["run_anyway"] = True
            print("it sets for new traininng")
        case_obj.train_model(**extra_args)
    # case_obj.validate_model()
    # exp_obj.validate_experiment()
    # exp_obj.visualize_report()


# TODO: chane this, this is garbage
def train_on_experiment_loop(args):
    start(args)
    experiment_name_train = args.experiment
    target_on_exp_to_train = experiment_name_train.split(":")[0]
    case_name_train = args.case
    for experiment_name, exp in experiments_dict.items():
        target_on_exp = experiment_name.split(":")[0]
        if target_on_exp_to_train != target_on_exp:
            continue
        print("Train on experiment loop, in the fokccer")
        print(experiment_name)
        # exp.setup()
        print("-------------------------------------------------------------")
        for case_obj in exp.conf:
            if experiment_name == experiment_name_train:
                case_name = str(case_obj.name)
                # if len(case_name.split("_")) > 4:
                #     continue
                # if "252" in case_obj.name:
                #     continue
                # if case_obj.name.endswith("_adam"):
                #     continue

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
    validation_target = getattr(args, "validation_target", None)
    if validation_target is not None:
        os.environ["VALIDATION_TARGET"] = validation_target

    name = getattr(args, "name", None)
    for experiment_name, exp in experiments_dict.items():
        # exp.setup()
        print("-------------------------------------------------------------")
        print(exp.name)
        print("Is complete?", exp.complete)
        # if exp.complete:
        # continue
        if name is not None:
            if name != experiment_name:
                continue

        print(exp.previous_experiment)
        # if exp.previous_experiment:
        #     exp.setup()
        print("Is complete?", exp.complete)

        if exp.complete:
            if not exp.validation_complete:
                exp.validate_experiment()
                exp.visualize_report()
            continue
        if not getattr(exp, "conf", None):
            print("does this?")
            exp.setup()
        for case_obj in exp.conf:
            case_name = str(case_obj.name)
            # case_obj.train_model()
            # case_obj.validate_model()
            run_training = not case_obj.complete or exp.final_experiment
            print("RUUU", run_training)
            print()
            # if case_obj.epochs>exp.epochs:
            #     run_training=False
            if run_training:
                #     command = (
                #         f"KERAS_BACKEND={case_obj.keras_backend}"
                #         f" muaddib train_on_experiment_loop --experiment={experiment_name} --case={case_name}"
                #     )
                command = (
                    f"KERAS_BACKEND={case_obj.keras_backend}"
                    f" muaddib train_on_call --experiment={experiment_name} --case={case_name}"
                )
                if validation_target is not None:
                    command += f" --validation_target={validation_target}"

                open_new_console(command)
            case_obj.validate_model()
        exp.validate_experiment()
        exp.visualize_report()
        print("exp.worthy_cases in commandsssssssssssssssss")
        print([f.name for f in exp.worthy_cases])

        print(exp.worthy_cases)
        print("best result:", f"{exp.name} with: {exp.best_result}")
        if exp.previous_experiment:
            print(
                "previous name result",
                f"{exp.previous_experiment.name} with: {exp.previous_experiment.best_result}",
            )
