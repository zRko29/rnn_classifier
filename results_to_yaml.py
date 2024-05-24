import os
import pandas as pd
from src.utils import read_yaml, extract_best_loss_from_event_file
from argparse import ArgumentParser
from src.utils import save_yaml


def get_loss_and_params(dir: str) -> pd.DataFrame:
    all_loss_hyperparams = []
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
                        "directory": int(directory.split("_")[-1]),
                        **loss_value,
                        **parameter_dict,
                    }
                )

    results = pd.DataFrame(all_loss_hyperparams)

    results = results.sort_values("directory")

    return results


def main(args):
    path = args.experiment_path

    results = get_loss_and_params(path)

    final_dict = {
        dir: other_values
        for dir, other_values in zip(
            results["directory"],
            results.drop("directory", axis=1).to_dict(orient="records"),
        )
    }

    save_yaml(final_dict, os.path.join(path, "results.yaml"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--experiment_path")
    args = parser.parse_args()

    main(args)
