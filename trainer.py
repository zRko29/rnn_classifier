from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional, Dict

if TYPE_CHECKING:
    from pytorch_lightning.callbacks import callbacks
    from src.mapping_helper import StandardMap

from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)

from src.data_helper import Data
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
            monitor=args.monitor_checkpoint,
            mode=args.mode_checkpoint,
            dirpath=save_path,
            filename="model",
            save_on_train_epoch_end=True,
        ),
        EarlyStopping(
            monitor=args.monitor_stopping,
            mode=args.mode_stopping,
            min_delta=5e-2,
            patience=10,
        ),
    ]


def main(
    args: Namespace,
    params: dict,
    map_object: Optional[StandardMap] = None,
) -> None:
    logger = logging.getLogger("rnn_classifier")

    datamodule_train: Data = Data(
        data_path="training_data",
        train_size=args.train_size,
        K=params.get("K_train"),
        binary=True,
        map_object=map_object,
        params=params,
    )

    datamodule_val: Data = Data(
        data_path="training_data",
        train_size=1.0,
        K=params.get("K_val"),
        binary=True,
        map_object=map_object,
        params=params,
    )

    tb_logger = TensorBoardLogger(save_dir="", name=args.path, default_hp_metric=False)

    save_path: str = os.path.join(tb_logger.name, f"version_{tb_logger.version}")

    trainer = Trainer(
        max_epochs=args.epochs,
        precision=params.get("precision"),
        logger=tb_logger,
        callbacks=get_callbacks(args, save_path),
        deterministic=False,
        benchmark=True,
        check_val_every_n_epoch=3,
        enable_progress_bar=args.progress_bar,
        devices=args.devices,
        num_nodes=args.num_nodes,
    )

    if trainer.is_global_zero:
        print(f"Running version_{tb_logger.version}.")
        logger.info(f"Running trainer.py (version_{tb_logger.version}).")

        print_args = args.__dict__.copy()
        del print_args["path"]
        logger.info(f"args = {print_args}")

    model = get_model(args, params)
    model.set_weight(datamodule_train.spectrum)

    # take train_dataloader in both
    trainer.fit(
        model,
        train_dataloaders=datamodule_train.train_dataloader(),
        val_dataloaders=datamodule_val.train_dataloader(),
    )


def get_model(args: Namespace, params: Dict) -> None:
    if params.get("rnn_type") == "vanillarnn":
        from src.VanillaRNN import Vanilla as Model
    elif params.get("rnn_type") == "mgu":
        from src.MGU import MGU as Model
    elif params.get("rnn_type") == "resrnn":
        from src.ResRNN import ResRNN as Model
    else:
        raise ValueError(f"Invalid rnn_type: {params.get('rnn_type')}")

    if args.checkpoint_path:
        model = Model.load_from_checkpoint(args.checkpoint_path, map_location="cpu")
    else:
        model = Model(**params)

    return model


if __name__ == "__main__":
    args: Namespace = import_parsed_args("Classifier trainer")
    args.path = os.path.abspath(args.path)

    logger = setup_logger(args.path, "rnn_classifier")

    params_path = os.path.join(args.path, "parameters.yaml")
    params = read_yaml(params_path)

    main(args, params)
