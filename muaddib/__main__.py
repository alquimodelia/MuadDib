# -*- coding: utf-8 -*-
import argparse
import logging
import logging.config

from muaddib.commands import (
    experiment,
    init,
    new,
    process_data,
    start,
    train,
    train_case,
    train_on_experiment_loop,
)

log = logging.getLogger(__name__)


def get_arg_parser():
    """
    Defines the arguments to this script by using Python's
        [argparse](https://docs.python.org/3/library/argparse.html)
    """
    parser = argparse.ArgumentParser(
        description="Beyond Mentalic - A Machine Learning Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-l",
        "--logconf",
        type=str,
        action="store",
        help="location of the logging config (yml) to use",
    )

    subparsers = parser.add_subparsers(dest="command", help="sub-command help")

    parser_new = subparsers.add_parser(
        "new", help="Create a new project based on a template"
    )
    parser_new.add_argument(
        "--template_name",
        choices=["mentalic", "renewable-energy-allocation"],
        help="Name of the template to use for the new project",
    )
    parser_new.add_argument("project_name", help="Name of the new project")

    subparsers.add_parser(
        "process_data", help="Create a new project based on a template"
    )

    parser_train = subparsers.add_parser("train", help="start help")
    parser_train.add_argument("--model_to_train", help="Model to train")
    parser_train.add_argument("--model_name", help="Model to train")
    parser_train.add_argument("--fit_args", help="Model to train")
    parser_train.add_argument("--compile_args", help="Model to train")

    parser_train_case = subparsers.add_parser("train_case", help="start help")
    parser_train_case.add_argument("--case", help="Model to train")
    parser_train_case.add_argument("--experiment", help="Model to train")

    parser_train_on_experiment_loop = subparsers.add_parser(
        "train_on_experiment_loop", help="start help"
    )
    parser_train_on_experiment_loop.add_argument(
        "--case", help="Model to train"
    )
    parser_train_on_experiment_loop.add_argument(
        "--experiment", help="Model to train"
    )

    subparsers.add_parser("init", help="start help")
    subparsers.add_parser("experiment", help="start help")

    subparsers.add_parser("start", help="start help")
    subparsers.add_parser("plots", help="plots help")
    subparsers.add_parser("run", help="run help")

    return parser


def enable_logging(args: argparse.Namespace):
    if args.logconf is None:
        return
    import yaml

    with open(args.logconf, "r") as yml_logconf:
        logging.config.dictConfig(
            yaml.load(yml_logconf, Loader=yaml.SafeLoader)
        )
    log.info(f"Logging enabled according to config in {args.logconf}")


def handle_new(args):
    print(f"Creating new project: {args.project_name}")


def handle_start():
    print("Starting...")


def handle_plots():
    print("Generating plots...")


def handle_run():
    print("Running...")


COMMANDS = {
    "new": new,
    "train": train,
    "init": init,
    "experiment": experiment,
    "train_case": train_case,
    "process_data": process_data,
    "start": start,
    "plots": handle_plots,
    "run": handle_run,
    "train_on_experiment_loop": train_on_experiment_loop,
}


def main():
    """The main entry point to this module."""
    args = get_arg_parser().parse_args()
    enable_logging(args)

    command_function = COMMANDS.get(args.command)
    if command_function is not None:
        command_function(args)
    else:
        print(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
