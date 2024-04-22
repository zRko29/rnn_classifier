import os
import pytorch_lightning as pl
from src.helper import Model, Data
from src.utils import read_yaml, get_inference_folders, plot_labeled_data
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
    version: Optional[int] = 71
    name: str = "overfitting_K=1.5"

    directory_path: str = f"logs/{name}"

    folders: List = get_inference_folders(directory_path, version)

    for log_path in folders:
        print(f"log_path: {log_path}")
        params_path: str = os.path.join(log_path, "hparams.yaml")
        params: dict = read_yaml(params_path)

        # NOTE: set these parameters to reduce loaded data
        params.update({"init_points": 350, "steps": 100})

        K = [1.5]

        for idx, K in enumerate(K):
            model_path: str = os.path.join(log_path, f"model.ckpt")
            model = Model(**params).load_from_checkpoint(model_path, map_location="cpu")

            datamodule = Data(
                data_path="training_data",
                K=K,
                train_size=1.0,
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

            print()
            print(
                f"{idx} (K = {K}): loss = {predictions['loss']:.2e}, accuracy = {predictions['accuracy']:.2f}, f1 = {predictions['f1']:.2f}"
            )

            plot_labeled_data(
                datamodule.thetas,
                datamodule.ps,
                datamodule.spectrum,
                title=f"True Labels (K={K})",
                save_path=f"{log_path}/true_K={K}",
            )

            plot_labeled_data(
                datamodule.thetas,
                datamodule.ps,
                predictions["predicted_labels"],
                f"Predicted Labels (K={K})",
                save_path=f"{log_path}/predicted_K={K}",
            )

        print()
        print("-----------------------------")


if __name__ == "__main__":
    main()
