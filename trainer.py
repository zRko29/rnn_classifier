from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from pytorch_lightning.callbacks import callbacks
    from src.mapping_helper import StandardMap

from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    DeviceStatsMonitor,
)

from src.helper import Model, Data
from src.utils import import_parsed_args, read_yaml, setup_logger

from argparse import Namespace
import os
import warnings
import logging

os.environ["GLOO_SOCKET_IFNAME"] = "en0"
warnings.filterwarnings(
    "ignore",
    module="pytorch_lightning",
)
logging.getLogger("pytorch_lightning").setLevel("INFO")
seed_everything(42, workers=True)


def get_callbacks(args: Namespace, save_path: str) -> List[callbacks]:
    return [
        ModelCheckpoint(
            monitor=args.monitor,
            mode=args.mode,
            dirpath=save_path,
            filename="model",
            save_on_train_epoch_end=True,
        ),
        EarlyStopping(
            monitor=args.monitor,
            mode=args.mode,
            min_delta=1e-6,
            patience=400,
        ),
        # DeviceStatsMonitor(cpu_stats=False),
    ]


def main(
    args: Namespace,
    params: dict,
    logger: Optional[logging.Logger],
    map_object: Optional[StandardMap] = None,
) -> None:

    datamodule: Data = Data(
        data_path="training_data",
        train_size=args.train_size,
        K=params.get("K"),
        binary=True,
        map_object=map_object,
        params=params,
    )

    model: Model = Model(**params)

    tb_logger = TensorBoardLogger(
        save_dir="", name=args.experiment_path, default_hp_metric=False
    )

    save_path: str = os.path.join(tb_logger.name, "version_" + str(tb_logger.version))

    trainer = Trainer(
        max_epochs=args.epochs,
        precision=params.get("precision"),
        logger=tb_logger,
        callbacks=get_callbacks(args, save_path),
        deterministic=True,
        enable_progress_bar=args.progress_bar,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        num_nodes=args.num_nodes,
    )

    if trainer.is_global_zero:
        logger.info(f"Running trainer.py.")

        print_args = args.__dict__.copy()
        del print_args["experiment_path"]
        logger.info(f"args = {print_args}")

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    args: Namespace = import_parsed_args("Autoregressor trainer")
    args.experiment_path = os.path.abspath(args.experiment_path)

    logger = setup_logger(args.experiment_path)

    params_path = os.path.join(args.experiment_path, "current_params.yaml")
    params = read_yaml(params_path)

    main(args, params, logger)
