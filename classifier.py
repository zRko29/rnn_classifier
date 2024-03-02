import os, yaml
import pytorch_lightning as pl

import warnings

warnings.filterwarnings(
    "ignore", ".*Consider increasing the value of the `num_workers` argument*"
)


import logging

logging.getLogger("pytorch_lightning").setLevel(0)

from utils.helper import Model, Data

if __name__ == "__main__":
    version: int = 12
    name: str = "classification_3"

    directory_path: str = f"logs/{name}"

    if version is not None:
        folders: list[str] = [os.path.join(directory_path, f"version_{str(version)}")]
    else:
        folders: list[str] = [
            os.path.join(directory_path, folder)
            for folder in os.listdir(directory_path)
            if os.path.isdir(os.path.join(directory_path, folder))
        ]
        folders.sort()

    for log_path in folders:
        print(f"log_path: {log_path}")
        params_path: str = os.path.join(log_path, "hparams.yaml")
        with open(params_path, "r") as file:
            params: dict = yaml.safe_load(file)

        params.update({"init_points": 30, "steps": 300})

        K_list: list = [0.1, 0.2, 0.3, 0.5, 1.0, 1.5]

        for idx, K in enumerate(K_list):
            datamodule: Data = Data(
                data_path="testing_data",
                K_list=[K],
                plot_data=True,
                binary=True,
                params=params,
                reduce_init_points=True,
            )

            model_path: str = os.path.join(log_path, f"model.ckpt")
            model: Model = Model(**params).load_from_checkpoint(model_path)

            trainer = pl.Trainer(
                precision=params.get("precision"),
                enable_progress_bar=False,
                logger=False,
            )
            predictions: dict = trainer.predict(model=model, dataloaders=datamodule)[0]
            loss = predictions["loss"]
            accuracy = predictions["accuracy"]
            f1 = predictions["f1"]
            predicted_labels = predictions["predicted_labels"]

            datamodule.plot_data(datamodule.thetas, datamodule.ps, predicted_labels)

            print(
                f"{idx} (K = {K}): loss = {loss:.2e}, accuracy = {accuracy:.2f}, f1 = {f1:.2f}"
            )
            print()

        print("-" * 30)
