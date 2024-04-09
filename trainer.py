from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from pytorch_lightning.callbacks import callbacks
    from src.mapping_helper import StandardMap

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    DeviceStatsMonitor,
)

from src.helper import Model, Data, Gridsearch, CustomCallback
from src.utils import import_parsed_args, read_yaml, setup_logger, measure_time

import os
import warnings
import logging
from argparse import Namespace
from time import sleep

os.environ["GLOO_SOCKET_IFNAME"] = "en0"
warnings.filterwarnings(
    "ignore",
    module="pytorch_lightning",
)
logging.getLogger("pytorch_lightning").setLevel("INFO")


def get_callbacks(save_path: str) -> List[callbacks]:
    return [
        ModelCheckpoint(
            monitor="loss/val",
            mode="min",
            dirpath=save_path,
            filename="model",
            save_on_train_epoch_end=True,
        ),
        EarlyStopping(
            monitor="loss/val",
            mode="min",
            min_delta=1e-5,
            patience=300,
        ),
        # DeviceStatsMonitor(),
        CustomCallback(),
    ]


@measure_time
def main(
    args: Namespace,
    params: dict,
    sleep_time: int,
    logger: Optional[logging.Logger],
    map_object: Optional[StandardMap] = None,
) -> None:
    sleep(sleep_time)

    datamodule: Data = Data(
        data_path="training_data",
        train_size=0.8,
        K=params.get("K"),
        binary=True,
        map_object=map_object,
        params=params,
    )

    model: Model = Model(**params)

    tb_logger = TensorBoardLogger(
        save_dir="",
        name=params.get("name"),
        default_hp_metric=False,
    )

    save_path: str = os.path.join(tb_logger.name, "version_" + str(tb_logger.version))

    logger.info(f"Started version_{tb_logger.version}")

    trainer = Trainer(
        max_epochs=params.get("epochs"),
        precision=params.get("precision"),
        logger=tb_logger,
        callbacks=get_callbacks(save_path),
        enable_progress_bar=args.progress_bar,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        num_nodes=args.num_nodes,
    )

    trainer.fit(model, datamodule)
    logger.info(f"Model trained and saved in '{save_path}'.")


if __name__ == "__main__":
    args: Namespace = import_parsed_args("Autoregressor trainer")

    args.params_dir = os.path.abspath(args.params_dir)

    params = read_yaml(args.params_dir)
    del params["gridsearch"]

    params["name"] = os.path.abspath(params["name"])

    logger = setup_logger(params["name"])
    logger.info("Started trainer.py")
    logger.info(f"{args.__dict__=}")

    run_time = main(args, params, 0, logger)

    logger.info(f"Finished trainer.py in {run_time}.\n")
