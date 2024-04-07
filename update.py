from typing import Dict, Tuple, List
import os
import pandas as pd
from argparse import Namespace
import logging

from src.utils import (
    read_yaml,
    import_parsed_args,
    setup_logger,
    measure_time,
    save_yaml,
    save_last_params,
    find_new_path,
    extract_best_loss_from_event_file,
)


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


def get_loss_and_params(dir: str, logger: logging.Logger) -> pd.DataFrame:
    all_loss_hyperparams = []
    try:
        for directory in sorted(os.listdir(dir)):
            loss_value = None
            parameter_dict = None
            if os.path.isdir(os.path.join(dir, directory)):
                for file in os.listdir(os.path.join(dir, directory)):
                    if "events" in file.split("."):
                        file_path = os.path.join(dir, directory, file)
                        loss_value = extract_best_loss_from_event_file(file_path)

                    elif file == "hparams.yaml":
                        file_path = os.path.join(dir, directory, file)
                        parameter_dict = read_yaml(file_path)

                if loss_value and parameter_dict:
                    all_loss_hyperparams.append({**loss_value, **parameter_dict})
    except FileNotFoundError as e:
        logger.error(e)
        raise e

    return pd.DataFrame(all_loss_hyperparams)


def compute_parameter_intervals(
    results: pd.DataFrame,
    args: Namespace,
    logger: logging.Logger,
) -> Dict[str, Tuple[float, float]]:
    gridsearch_params = read_yaml(args.params_dir)["gridsearch"]

    parameters = []
    for column in results.columns:
        if (
            len(results[column].unique()) > 1 or column in gridsearch_params.keys()
        ) and column != "best_loss":
            try:
                type = gridsearch_params[column]["type"]
            except KeyError:
                logger.warning(
                    f"Variable parameter '{column}' is not included in the gridsearch parameters."
                )
                continue
            if type:
                parameters.append(Parameter(name=column, type=type))

    # don't filter before because a parmeter could have the same value for all "good" rows
    # don't filter after because you wouldn't get optimal intervals
    try:
        results = results[results["best_loss"] < args.max_loss]
    except KeyError:
        logger.warning("There are probably no results in folder.")
        return None

    if len(results) < args.min_good_samples:
        logger.warning(
            f"Found {len(results)} (< {args.min_good_samples}) good samples. Parameters will not be updated."
        )
        return None
    else:
        logger.info(
            f"Found {len(results)} (> {args.min_good_samples}) good samples. Parameters will be updated."
        )

    for param in parameters:
        if param.type == "float":
            param.min = results[param.name].min()
            param.max = results[param.name].max()

        elif param.type == "int":
            param.min = results[param.name].min()
            param.max = results[param.name].max()

        elif param.type == "choice":
            param.value_counts = results[param.name].value_counts().to_dict()
            param.count = results[param.name].count()

            # only keep "good" values in list
            dict_copy = param.value_counts.copy()
            for key, value_count in param.value_counts.items():
                if value_count < param.count * 1 / 5 and len(dict_copy) > 1:
                    del dict_copy[key]
            param.value_counts = list(dict_copy.keys())

    return parameters


def update_yaml_file(
    args: Namespace, events_dir: str, parameters: List[Parameter]
) -> None:
    if parameters is not None:
        gridsearch_dict = {}
        for param in parameters:
            if param.type in ["float", "int"]:
                gridsearch_dict[param.name] = {
                    "lower": param.min,
                    "upper": param.max,
                    "type": param.type,
                }
            elif param.type == "choice":
                gridsearch_dict[param.name] = {
                    "list": param.value_counts,
                    "type": param.type,
                }

        yaml_params = read_yaml(args.params_dir)

        # specify new folder
        yaml_params["name"] = find_new_path(events_dir)

        # update gridsearch parameters
        yaml_params["gridsearch"] = gridsearch_dict

        save_yaml(yaml_params, args.params_dir)
        save_last_params(yaml_params, events_dir)


@measure_time
def main(args: Namespace, logger: logging.Logger) -> None:
    events_dir = read_yaml(args.params_dir)["name"]

    loss_and_params = get_loss_and_params(events_dir, logger)
    parameters = compute_parameter_intervals(
        results=loss_and_params, args=args, logger=logger
    )

    update_yaml_file(args, events_dir, parameters)


if __name__ == "__main__":
    args: Namespace = import_parsed_args("Parameter updater")

    params = read_yaml(args.params_dir)

    logs_dir = args.logs_dir or params["name"]

    logger = setup_logger(logs_dir)
    logger.info("Started update.py")
    logger.info(f"{args.__dict__=}")

    run_time = main(args, logger)

    logger.info(f"Finished trainer.py in {run_time}.\n")
