from typing import List
import numpy as np
import yaml
from argparse import Namespace, ArgumentParser
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch
import matplotlib.pyplot as plt

import logging


def read_yaml(parameters_path: str) -> dict:
    with open(parameters_path, "r") as file:
        return yaml.safe_load(file)


def save_yaml(file: dict, param_file_path: str) -> dict[str | float | int]:
    with open(param_file_path, "w") as f:
        yaml.dump(file, f, default_flow_style=None, default_style=None)


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
    logger = logging.getLogger("rnn_classifier")
    logger.setLevel(logging.INFO)

    try:
        os.makedirs(log_file_path)
    except FileExistsError:
        pass

    log_file_name = os.path.join(log_file_path, "logs.log")
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


class Parameter:
    def __init__(self, name: str, type: str) -> None:
        self.name = name
        self.type = type

        if self.type in ["float", "int"]:
            self.min = float("inf")
            self.max = -float("inf")

        elif self.type == "choice":
            self.value_counts = {}
            self.count = 0


def read_events_file(events_file_path: str) -> EventAccumulator:
    event_acc = EventAccumulator(events_file_path)
    event_acc.Reload()
    return event_acc


def extract_best_loss_from_event_file(events_file_path: str) -> str | float | int:
    event_values = read_events_file(events_file_path)
    for tag in event_values.Tags()["scalars"]:
        if tag == "best_loss":
            return {"best_loss": event_values.Scalars(tag)[-1].value}


class Gridsearch:
    def __init__(self, params_path: str, use_defaults: bool = False) -> None:
        self.path = params_path
        self.use_defaults = use_defaults

    def update_params(self) -> dict:
        params = read_yaml(self.path)
        if not self.use_defaults:
            params = self._update_params(params)

        try:
            del params["gridsearch"]
        except KeyError:
            pass

        return params

    def _update_params(self, params) -> dict:
        # don't use any seed
        rng: np.random.Generator = np.random.default_rng(None)

        for key, space in params.get("gridsearch").items():
            type = space.get("type")
            if type == "int":
                params[key] = int(rng.integers(space["lower"], space["upper"] + 1))
            elif type == "choice":
                list = space.get("list")
                choice = rng.choice(list)
                try:
                    choice = float(choice)
                except:
                    choice = str(choice)
                params[key] = choice
            elif type == "float":
                params[key] = rng.uniform(space["lower"], space["upper"])

        return params


def plot_labeled_data(thetas: np.ndarray, ps: np.ndarray, spectrum: np.ndarray) -> None:
    plt.figure(figsize=(7, 4))
    chaotic_indices = np.where(np.array(spectrum) == 1)[0]
    regular_indices = np.where(np.array(spectrum) == 0)[0]
    plt.plot(
        thetas[:, chaotic_indices],
        ps[:, chaotic_indices],
        "ro",
        markersize=0.5,
    )
    plt.plot(
        thetas[:, regular_indices],
        ps[:, regular_indices],
        "bo",
        markersize=0.5,
    )
    legend_handles = [
        plt.scatter([], [], color="red", marker=".", label="Chaotic"),
        plt.scatter([], [], color="blue", marker=".", label="Regular"),
    ]
    plt.legend(handles=legend_handles)
    plt.xlabel(r"$\theta$")
    plt.ylabel("p")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.show()


def import_parsed_args(script_name: str) -> Namespace:
    parser = ArgumentParser(prog=script_name)

    parser.add_argument(
        "--experiment_path",
        type=str,
        default="logs/",
        help="Path to the experiment directory. (default: %(default)s)",
    )

    if script_name == "Gridsearch step":
        parser.add_argument(
            "--default_params",
            "-dflt",
            action="store_true",
            help="Use default parameters for the gridsearch. (default: False)",
        )

    elif script_name == "Autoregressor trainer":
        parser.add_argument(
            "--epochs",
            type=int,
            default=1000,
            help="Number of epochs to train the model for. (default: %(default)s)",
        )
        parser.add_argument(
            "--monitor",
            type=str,
            default="loss/train",
            help="Metric to monitor for early stopping and checkpointing. (default: %(default)s)",
        )
        parser.add_argument(
            "--mode",
            type=str,
            default="min",
            help="Mode (min/max) for early stopping and checkpointing. (default: %(default)s)",
        )
        parser.add_argument(
            "--train_size",
            type=float,
            default=0.8,
            help="Fraction of data to use for training. (default: %(default)s)",
        )
        parser.add_argument(
            "--progress_bar",
            "-prog",
            action="store_true",
            help="Show progress bar during training. (default: False)",
        )
        parser.add_argument(
            "--accelerator",
            type=str,
            default="auto",
            choices=["auto", "cpu", "gpu"],
            help="Specify the accelerator to use. (default: %(default)s)",
        )
        parser.add_argument(
            "--devices",
            nargs="*",
            type=int,
            help="List of devices to use. (default: %(default)s)",
        )
        parser.add_argument(
            "--strategy",
            type=str,
            default="auto",
            help="Specify the training strategy. (default: %(default)s)",
        )
        parser.add_argument(
            "--num_nodes",
            type=int,
            default=1,
            help="Specify number of nodes to use. (default: 1)",
        )

    elif script_name == "Parameter updater":
        parser.add_argument(
            "--max_good_loss",
            type=float,
            default=5e-6,
            help="Maximum loss value considered acceptable for selecting parameters. (default: %(default)s)",
        )
        parser.add_argument(
            "--min_good_samples",
            type=int,
            default=3,
            help="Minimum number of good samples required to start updating parameters. (default: %(default)s)",
        )
        parser.add_argument(
            "--check_every_n_steps",
            type=int,
            default=1,
            help="Check for new good samples every n steps. Its suggested that check_every_n_steps < min_good_samples, so that results are less likely to converge to a local optimium. (default: %(default)s)",
        )
        parser.add_argument(
            "--current_step",
            type=int,
            default=1,
            help="Current step of the training. (default: None)",
        )

    return parser.parse_args()
