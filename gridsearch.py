from argparse import Namespace
import os
from src.utils import (
    import_parsed_args,
    save_yaml,
    setup_logger,
    Gridsearch,
)


def main(args: Namespace) -> None:
    params_path = os.path.join(args.path, "parameters.yaml")
    gridsearch = Gridsearch(params_path, use_defaults=args.default_params)
    updated_params = gridsearch.update_params()

    save_yaml(updated_params, os.path.join(args.path, "current_params.yaml"))


if __name__ == "__main__":
    args: Namespace = import_parsed_args("Gridsearch step")

    args.path = os.path.abspath(args.path)

    logger = setup_logger(args.path, "rnn_classifier")
    logger.info("Running gridsearch.py")

    print_args = args.__dict__.copy()
    del print_args["path"]
    logger.info(f"args = {print_args}")

    main(args)
