import os, yaml
from pytorch_lightning import Trainer

import warnings
import logging

warnings.filterwarnings(
    "ignore",
    module="pytorch_lightning",
)
logging.getLogger("pytorch_lightning").setLevel(0)

from src.helper import Model, Data
from src.utils import read_yaml, get_inference_folders
from typing import List


def main() -> None:
    version: int = 0
    name: str = "overfitting_K=0.1/0"

    directory_path: str = f"logs/{name}"

    folders: List = get_inference_folders(directory_path, version)

    for log_path in folders:
        print(f"log_path: {log_path}")
        params_path: str = os.path.join(log_path, "hparams.yaml")
        params: dict = read_yaml(params_path)

        params.update({"init_points": 30, "steps": 300})

        K = [0.1, 0.2, 0.3, 0.5, 1.0, 1.5]

        for idx, K in enumerate(K):
            datamodule: Data = Data(
                data_path="testing_data",
                K=K,
                plot_data=True,
                binary=True,
                reduce_init_points=True,
                params=params,
            )

            model_path: str = os.path.join(log_path, f"model.ckpt")
            model = Model(**params).load_from_checkpoint(model_path)

            trainer = Trainer(
                precision=params.get("precision"),
                enable_progress_bar=False,
                logger=False,
            )

            predictions: dict = trainer.predict(model=model, dataloaders=datamodule)[0]
            predicted_labels = predictions["predicted_labels"]

            datamodule.plot_data(datamodule.thetas, datamodule.ps, predicted_labels)

            print(
                f"{idx} (K = {K}): loss = {predictions['loss']:.2e}, accuracy = {predictions['accuracy']:.2f}, f1 = {predictions['f1']:.2f}"
            )
            print()

        print("----------------------")


if __name__ == "__main__":
    main()
