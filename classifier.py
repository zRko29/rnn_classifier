import os
import torch
import pytorch_lightning as pl
from src.data_helper import Data
from src.utils import (
    read_yaml,
    get_inference_folders,
    plot_labeled_data,
    subplot_labeled_data,
)
from typing import Optional, Dict, Tuple
from argparse import ArgumentParser, Namespace

import warnings
import logging

from src.mapping_helper import StandardMap

warnings.filterwarnings(
    "ignore",
    module="pytorch_lightning",
)
logging.getLogger("pytorch_lightning").setLevel(0)
pl.seed_everything(42, workers=True)


def main(args: Namespace) -> None:
    version: Optional[int] = args.version or None
    directory_path: str = "logs/paper/"
    # directory_path: str = "logs/improve_paper/"

    folders = get_inference_folders(directory_path, version)

    for log_path in folders:
        print(f"log_path: {log_path}")
        params_path: str = os.path.join(log_path, "hparams.yaml")
        params: dict = read_yaml(params_path)

        params_update: Dict = {}
        params_update.update({"seq_len": 20})
        params_update.update({"K": 1.5})
        # params_update.update({"K": [3.0, 3.1, 3.2, 3.3, 3.4, 3.5]})
        params_update.update({"init_points": 200})
        params_update.update({"steps": 100})

        map_object = None
        # map_object = StandardMap(
        #     init_points=100, steps=60, K=1.5, seed=42, sampling="random"
        # )

        data = "testing_data"
        # data = "training_data"

        predictions, datamodule = inference(
            log_path, params, params_update, map_object=map_object, data=data
        )

        K = params_update.get("K") or params.get("K")

        print(
            f"rnn_type={params.get('rnn_type')}, {K = }: loss = {predictions['loss']:.2e}, accuracy = {predictions['accuracy']:.2f}, f1 = {predictions['f1']:.2f}, precision = {predictions['precision']:.2f}, recall = {predictions['recall']:.2f}, specificity = {predictions['specificity']:.2f}, balanced_accuracy = {predictions['balanced_accuracy']:.2f}"
        )
        print()
        subplot_labeled_data(
            datamodule.thetas_orig,
            datamodule.ps_orig,
            [datamodule.spectrum, predictions["predicted_labels"]],
            ["Resnica", f"Napoved (UT = {predictions['balanced_accuracy']:.2f})"],
            save_path=f"{log_path}/results",
        )


def inference(
    log_path: str,
    params: Dict,
    params_update: Dict = None,
    map_object: StandardMap = None,
    data: str = "training_data",
) -> Tuple[Dict, Data]:
    # NOTE: set these parameters to reduce loaded data
    if params_update is not None:
        params.update(params_update)

    if params.get("rnn_type") == "vanillarnn":
        from src.VanillaRNN import Vanilla as Model
    elif params.get("rnn_type") == "mgu":
        from src.MGU import MGU as Model
    elif params.get("rnn_type") == "resrnn":
        from src.ResRNN import ResRNN as Model

    model_path: str = os.path.join(log_path, f"model.ckpt")
    model = Model(**params).load_from_checkpoint(model_path, map_location="cpu")
    model.eval()

    datamodule = Data(
        data_path=data,
        map_object=map_object,
        K=params.get("K"),
        train_size=1.0,
        binary=True,
        reduce_init_points=True,
        params=params,
    )

    trainer = pl.Trainer(
        precision=params.get("precision"),
        enable_progress_bar=True,
        logger=False,
        deterministic=True,
    )

    predictions: dict = trainer.predict(model=model, dataloaders=datamodule)[0]

    return predictions, datamodule


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--version", "-v", nargs="*", type=int, default=None)
    args = parser.parse_args()

    main(args)
