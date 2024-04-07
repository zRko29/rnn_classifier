from argparse import Namespace
from joblib import Parallel, delayed
import logging
import os

from trainer import main as train
from update import main as update

from src.helper import Gridsearch
from src.mapping_helper import StandardMap
from src.utils import measure_time, read_yaml, import_parsed_args, setup_logger


@measure_time
def main(args: Namespace, logger: logging.Logger) -> None:
    gridsearch = Gridsearch(args.params_dir, use_defaults=False)
    sleep_list = range(0, args.models_per_step * 5, 5)

    for i in range(args.optimization_steps):

        print()
        print(f"Started optimization step: {i + 1} / {args.optimization_steps}")
        logger.info(f"Started optimization step: {i + 1} / {args.optimization_steps}")
        print()

        try:
            Parallel(
                backend="loky",
                n_jobs=args.models_per_step,
                verbose=11,
            )(
                delayed(train)(args, params, sleep, logger)
                for sleep, params in zip(sleep_list, gridsearch)
            )
        except Exception as e:
            logger.error(e)
            raise e

        update(args, logger)
        print()
        print(f"Finished optimization step: {i + 1} / {args.optimization_steps}")
        logger.info(f"Finished optimization step: {i + 1} / {args.optimization_steps}")
        print()
        print("-----------------------------")


if __name__ == "__main__":
    args: Namespace = import_parsed_args("Hyperparameter optimizer")

    args.params_dir = os.path.abspath(args.params_dir)

    params = read_yaml(args.params_dir)
    del params["gridsearch"]

    params["name"] = os.path.abspath(params["name"])

    logger = setup_logger(params["name"])
    logger.info("Started optimize.py")
    logger.info(f"{args.__dict__=}")

    run_time = main(args, logger)

    logger.info(f"Finished optimize.py in {run_time}.\n")
