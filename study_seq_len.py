import os
import pytorch_lightning as pl
from src.helper import Model, Data
from src.utils import read_yaml, get_inference_folders, plot_f1_scores
from typing import List, Optional

import warnings
import logging
import numpy as np
import pyprind

warnings.filterwarnings(
    "ignore",
    module="pytorch_lightning",
)
logging.getLogger("pytorch_lightning").setLevel(0)
pl.seed_everything(42, workers=True)


def main() -> None:
    version: Optional[int] = 71
    directory_path: str = "../classifier_backup_1/overfitting_K=1.5"

    folders: List = get_inference_folders(directory_path, version)

    for log_path in folders:
        print(f"log_path: {log_path}")
        params_path: str = os.path.join(log_path, "hparams.yaml")
        params: dict = read_yaml(params_path)

        seq_lens = np.arange(10, 999, 5)
        f1_scores = []

        K = [1.5]

        for idx, K in enumerate(K):

            pbar = pyprind.ProgBar(
                iterations=len(seq_lens),
                bar_char="█",
                width=30,
                track_time=False,
                title="Computing f1 scores...",
            )

            for seq_len in seq_lens:

                # NOTE: set these parameters to reduce loaded data
                params.update({"init_points": 350, "seq_len": seq_len})

                model_path: str = os.path.join(log_path, f"model.ckpt")
                model = Model(**params).load_from_checkpoint(
                    model_path, map_location="cpu"
                )

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

                predictions: dict = trainer.predict(
                    model=model, dataloaders=datamodule
                )[0]

                f1_scores.append(predictions["f1"])

                pbar.update()

            plot_f1_scores(
                seq_lens, f1_scores, K, save_path=f"{log_path}/f1_scores_{K=}"
            )


if __name__ == "__main__":
    main()
