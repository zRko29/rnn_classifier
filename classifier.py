import os, yaml
import pytorch_lightning as pl

import warnings

warnings.filterwarnings(
    "ignore", ".*Consider increasing the value of the `num_workers` argument*"
)

import logging

logging.getLogger("pytorch_lightning").setLevel(0)

from utils.mapping_helper import StandardMap
from utils.helper import Model, Data

if __name__ == "__main__":
    version = None
    name = "classification_1"

    directory_path = f"logs/{name}"

    if version is not None:
        folders = [os.path.join(directory_path, f"version_{str(version)}")]
    else:
        folders = [
            os.path.join(directory_path, folder)
            for folder in os.listdir(directory_path)
            if os.path.isdir(os.path.join(directory_path, folder))
        ]
        folders.sort()

    for log_path in folders:
        print(f"log_path: {log_path}")
        params_path = os.path.join(log_path, "hparams.yaml")
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)

        params.update({"init_points": 30, "steps": 60, "sampling": "random"})
        maps = [
            StandardMap(K=0.1, params=params, seed=42),
            StandardMap(K=0.1, params=params, seed=41),
            StandardMap(K=0.1, params=params, seed=40),
            StandardMap(K=0.1, params=params, seed=39),
        ]

        for idx, map in enumerate(maps):
            datamodule = Data(
                map_object=map,
                train_size=1.0,
                plot_data=False,
                print_split=False,
                binary=True,
                params=params,
            )

            model_path = os.path.join(log_path, f"model.ckpt")
            model = Model(**params).load_from_checkpoint(model_path)

            trainer = pl.Trainer(
                enable_progress_bar=False,
                logger=False,
            )
            predictions = trainer.predict(model=model, dataloaders=datamodule)[0]
            loss = predictions["loss"]
            accuracy = predictions["accuracy"]

            print(f"{idx}: loss= {loss:.2e}, accuracy= {accuracy:.2f}")

        print("-" * 30)
