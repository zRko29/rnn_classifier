from typing import Dict, Tuple, List
import os
import pandas as pd
from argparse import Namespace
import logging

from src.utils import (
    import_parsed_args,
    read_yaml,
    save_yaml,
    setup_logger,
    extract_best_loss_from_event_file,
    Parameter,
)


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
                    all_loss_hyperparams.append(
                        {
                            "directory": directory,
                            **loss_value,
                            **parameter_dict,
                        }
                    )
    except FileNotFoundError as e:
        logger.error(e)
        raise e

    return pd.DataFrame(all_loss_hyperparams)


def compute_new_parameter_intervals(
    results: pd.DataFrame,
    args: Namespace,
    logger: logging.Logger,
    params_path: str,
) -> Dict[str, Tuple[float, float]]:
    gridsearch_params = read_yaml(params_path)["gridsearch"]

    parameters = []
    for column in results.columns:
        if (
            (len(results[column].unique()) > 1 or column in gridsearch_params.keys())
            and column != "best_loss"
            and column != "directory"
        ):
            try:
                type = gridsearch_params[column]["type"]
            except KeyError:
                logger.warning(
                    f"Parameter '{column}' changed during training but it is not included in the gridsearch parameters."
                )
                continue
            if type:
                parameters.append(Parameter(name=column, type=type))

    # don't filter before because a parmeter could have the same value for all "good" rows
    # don't filter after because you wouldn't get optimal intervals
    try:
        results = results[results["best_loss"] < args.max_good_loss]
    except KeyError:
        logger.warning("There are probably no results in folder.")
        return None

    if len(results) < args.min_good_samples:
        logger.warning(
            f"Found {len(results)} (< {args.min_good_samples}) good samples. Parameters will not be updated."
        )
        return None
    else:
        logger.warning(
            f"Found {len(results)} (>= {args.min_good_samples}) good samples. Updating parameters."
        )

    # ensures newer results at the bottom
    results = results.sort_values("directory", ascending=True)

    # NOTE: ensures that older results, which set larger intervals, get deprecated, so that newer results can set smaller intervals
    results = results.tail(args.min_good_samples)

    parameters = get_new_intervals(results, parameters)

    return parameters


def get_new_intervals(
    results: pd.DataFrame,
    parameters: List[Parameter],
) -> List[Parameter]:
    for param in parameters:
        if param.type == "float":
            param.min = float(results[param.name].min())
            param.max = float(results[param.name].max())

        elif param.type == "int":
            param.min = int(results[param.name].min())
            param.max = int(results[param.name].max())

        elif param.type == "choice":
            param.value_counts = results[param.name].value_counts().to_dict()
            param.count = results[param.name].count()

            # only keep "good" values in list
            dict_copy = param.value_counts.copy()
            for key, value_count in param.value_counts.items():
                if value_count < param.count * 1 / 5 and len(dict_copy) > 1:
                    del dict_copy[key]
            param.value_counts = list(dict_copy.keys())


def update_yaml_file(
    params_path: str,
    parameters: List[Parameter],
    logger: logging.Logger,
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

        yaml_params = read_yaml(params_path)

        if yaml_params.get("gridsearch") == gridsearch_dict:
            logger.info("Parameter intervals remained the same.")
        else:
            logger.info("Parameter intervals changed.")

        # update gridsearch parameters
        yaml_params["gridsearch"] = gridsearch_dict

        save_yaml(yaml_params, params_path)


def main(args: Namespace, logger: logging.Logger) -> None:
    params_path = os.path.join(args.experiment_path, "parameters.yaml")

    loss_and_params = get_loss_and_params(args.experiment_path, logger)

    parameters = compute_new_parameter_intervals(
        results=loss_and_params,
        args=args,
        logger=logger,
        params_path=params_path,
    )

    update_yaml_file(params_path, parameters, logger)


if __name__ == "__main__":
    args: Namespace = import_parsed_args("Parameter updater")
    args.experiment_path = os.path.abspath(args.experiment_path)

    logger = setup_logger(args.experiment_path)
    logger.info("Running update.py")

    if args.current_step % args.check_every_n_steps != 0:
        logger.warning("Skipping parameter update this step.")
        exit()

    print_args = args.__dict__.copy()
    del print_args["experiment_path"]
    logger.info(f"args = {print_args}")

    params_path = os.path.join(args.experiment_path, "parameters.yaml")
    params = read_yaml(params_path)

    main(args, logger)
