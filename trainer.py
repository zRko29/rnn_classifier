# import os
# from google.colab import drive

# drive.mount('/content/drive')
# os.chdir("/content/drive/My Drive/Work")

import pytorch_lightning as pl
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler
import os

from utils.helper import Model, Data, Gridsearch, CustomCallback

import warnings

warnings.filterwarnings(
    "ignore", ".*Consider increasing the value of the `num_workers` argument*"
)
warnings.filterwarnings(
    "ignore",
    ".*The number of training batches*",
)

import logging

logging.getLogger("pytorch_lightning").setLevel(0)

ROOT_DIR = os.getcwd()
CONFIG_DIR = os.path.join(ROOT_DIR, "config")


if __name__ == "__main__":
    # necessary to continue training from checkpoint, else set to None
    version = None
    name = "classification_1"
    num_vertices = 5

    gridsearch = Gridsearch(CONFIG_DIR, num_vertices)

    for _ in range(num_vertices):
        params = gridsearch.get_params()

        datamodule = Data(
            data_path="data",
            K_upper_lim=params.get("K_upper_lim"),
            train_size=1.0,
            plot_data=False,
            print_split=False,
            binary=False,
            params=params,
        )

        model = Model(**params)

        logs_path = "logs"

        # **************** callbacks ****************

        tb_logger = TensorBoardLogger(logs_path, name=name, default_hp_metric=False)

        save_path = os.path.join(logs_path, name, "version_" + str(tb_logger.version))

        print(f"Running version_{tb_logger.version}")
        print()

        checkpoint_callback = callbacks.ModelCheckpoint(
            monitor="acc/train",  # careful
            mode="max",
            dirpath=save_path,
            filename="model",
            save_on_train_epoch_end=True,
            save_top_k=1,
            verbose=False,
        )

        early_stopping_callback = callbacks.EarlyStopping(
            monitor="acc/val",
            mode="max",
            min_delta=1e-4,
            check_on_train_epoch_end=True,
            patience=30,
            verbose=False,
        )

        gradient_avg_callback = callbacks.StochasticWeightAveraging(swa_lrs=1e-3)

        progress_bar_callback = callbacks.TQDMProgressBar(refresh_rate=10)

        profiler_callback = SimpleProfiler(
            dirpath=save_path, filename="profiler_report"
        )

        # **************** trainer ****************

        trainer = pl.Trainer(
            profiler=profiler_callback,
            max_epochs=params.get("epochs"),
            precision=params.get("precision"),
            enable_progress_bar=True,
            logger=tb_logger,
            callbacks=[
                checkpoint_callback,
                # early_stopping_callback,
                progress_bar_callback,
                # gradient_avg_callback,
                CustomCallback(),
            ],
        )

        trainer.fit(model=model, datamodule=datamodule, ckpt_path=None)
