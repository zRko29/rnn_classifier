import os
import pytorch_lightning as pl
from src.helper import Model, Data
from src.utils import read_yaml, get_inference_folders
from typing import List, Optional

import warnings
import logging

warnings.filterwarnings(
    "ignore",
    module="pytorch_lightning",
)
logging.getLogger("pytorch_lightning").setLevel(0)
pl.seed_everything(42, workers=True)


def main() -> None:
    version: Optional[int] = 0
    name: str = "overfitting_K=0.1"

    directory_path: str = f"logs/{name}"

    folders: List = get_inference_folders(directory_path, version)

    for log_path in folders:
        print(f"log_path: {log_path}")
        params_path: str = os.path.join(log_path, "hparams.yaml")
        params: dict = read_yaml(params_path)

        # NOTE: set these parameters to reduce loaded data
        params.update({"init_points": 30, "steps": 300})

        K = [0.5, 1.0, 1.5]

        for idx, K in enumerate(K):
            model_path: str = os.path.join(log_path, f"model.ckpt")
            model = Model(**params).load_from_checkpoint(model_path, map_location="cpu")

            datamodule = Data(
                data_path="testing_data",
                K=K,
                train_size=1.0,
                plot_data=True,
                binary=True,
                reduce_init_points=True,
                params=params,
            )

            trainer = pl.Trainer(
                precision=params.get("precision"),
                enable_progress_bar=False,
                logger=False,
                deterministic=True,
            )

            predictions: dict = trainer.predict(model=model, dataloaders=datamodule)[0]

            datamodule.plot_labeled_data(
                datamodule.thetas,
                datamodule.ps,
                predictions["predicted_labels"],
            )

            print(
                f"{idx} (K = {K}): loss = {predictions['loss']:.2e}, accuracy = {predictions['accuracy']:.2f}, f1 = {predictions['f1']:.2f}"
            )
            print()

        print("-----------------------------")


if __name__ == "__main__":
    main()
