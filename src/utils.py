import time
from datetime import timedelta
from typing import Callable, List
import yaml
from argparse import Namespace, ArgumentParser
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import logging


def read_yaml(parameters_path: str) -> dict:
    with open(parameters_path, "r") as file:
        return yaml.safe_load(file)


def measure_time(func: Callable) -> Callable:
    """
    A decorator that measures the time a function takes to run.
    """

    def wrapper(*args, **kwargs):
        t1 = time.time()
        val = func(*args, **kwargs)
        t2 = timedelta(seconds=time.time() - t1)
        if val == None:
            return t2
        return val

    return wrapper


def get_inference_folders(directory_path: str, version: str) -> List[str]:
    if version is not None:
        folders: List[str] = [os.path.join(directory_path, f"version_{version}")]
    else:
        folders: List[str] = [
            os.path.join(directory_path, folder)
            for folder in os.listdir(directory_path)
            if os.path.isdir(os.path.join(directory_path, folder))
        ]
        folders.sort()
    return folders


def setup_logger(log_file_path: str) -> logging.Logger:
    logger = logging.getLogger("rnn_autoregressor")
    logger.setLevel(logging.INFO)

    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path)

    log_file_name = os.path.join(log_file_path, "logs.log")
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def save_last_params(yaml_params: dict, events_dir: str) -> None:
    folder = "/".join(events_dir.split("/")[:-1])
    save_yaml(yaml_params, os.path.join(folder, "last_parameters.yaml"))


def find_new_path(file_dir: str) -> str:
    path_split = file_dir.split("/")
    path_split[-1] = str(int(path_split[-1]) + 1)
    new_path = "/".join(path_split)
    try:
        os.mkdir(new_path)
    except FileExistsError:
        pass
    return new_path


def save_yaml(file: dict, param_file_path: str) -> dict[str | float | int]:
    with open(param_file_path, "w") as f:
        yaml.safe_dump(file, f, default_flow_style=None, default_style=None)


def read_events_file(events_file_path: str) -> EventAccumulator:
    event_acc = EventAccumulator(events_file_path)
    event_acc.Reload()
    return event_acc


def extract_best_loss_from_event_file(events_file_path: str) -> str | float | int:
    event_values = read_events_file(events_file_path)
    for tag in event_values.Tags()["scalars"]:
        if tag == "metrics/min_train_loss":
            return {"best_loss": event_values.Scalars(tag)[-1].value}


def import_parsed_args(script_name: str) -> Namespace:
    parser = ArgumentParser(prog=script_name)

    parser.add_argument(
        "--params_dir",
        type=str,
        default="config/parameters.yaml",
        help="Directory containing parameter files. (default: %(default)s)",
    )

    parser.add_argument(
        "--logs_dir",
        type=str,
        default=None,
        help="File containing logs. (default: folder where trainer outputs are saved)",
    )

    if script_name in ["Autoregressor trainer", "Hyperparameter optimizer"]:
        parser.add_argument(
            "--progress_bar",
            "-prog",
            action="store_true",
            help="Show progress bar during training. (default: False)",
        )
        parser.add_argument(
            "--accelerator",
            "-acc",
            type=str,
            default="auto",
            choices=["auto", "cpu", "gpu"],
            help="Specify the accelerator to use. Choices are 'auto', 'cpu', or 'gpu'. (default: %(default)s)",
        )
        parser.add_argument(
            "--num_devices",
            default="auto",
            help="Number of devices to use. (default: %(default)s)",
        )
        parser.add_argument(
            "--strategy",
            type=str,
            default="auto",
            choices=["auto", "ddp", "ddp_spawn"],
            help="Specify the training strategy. Choices are 'auto', 'ddp', or 'ddp_spawn'. (default: %(default)s)",
        )
        parser.add_argument(
            "--num_nodes",
            type=int,
            default=1,
            help="Specify number of nodes to use. (default: 1)",
        )

    if script_name in ["Parameter updater", "Hyperparameter optimizer"]:
        parser.add_argument(
            "--max_loss",
            type=float,
            default=5e-6,
            help="Maximum loss value considered acceptable for selecting parameters. (default: %(default)s)",
        )
        parser.add_argument(
            "--min_good_samples",
            type=int,
            default=3,
            help="Minimum number of good samples required for parameter selection, otherwise parameters aren't updated, but training continues. (default: %(default)s)",
        )

    if script_name == "Hyperparameter optimizer":
        parser.add_argument(
            "--optimization_steps",
            type=int,
            default=5,
            help="Number of optimization steps to perform. (default: %(default)s)",
        )
        parser.add_argument(
            "--models_per_step",
            type=int,
            default=5,
            help="Number of models to train in each optimization step. (default: %(default)s)",
        )

    return parser.parse_args()
