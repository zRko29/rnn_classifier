import torch.optim as optim
import pytorch_lightning as pl
import torch
import torchmetrics
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import timedelta
import os, yaml
from typing import Tuple, List

from src.utils import read_yaml


class Model(pl.LightningModule):
    def __init__(
        self,
        **params: dict,
    ):
        super(Model, self).__init__()
        self.save_hyperparameters()

        self.num_rnn_layers: int = params.get("num_rnn_layers")
        self.num_lin_layers: int = params.get("num_lin_layers")
        dropout: float = params.get("dropout")
        self.lr: float = params.get("lr")
        self.optimizer: str = params.get("optimizer")

        # ----------------------
        # NOTE: This logic is kept so that variable layer sizes can be reimplemented in the future
        rnn_layer_size: int = params.get("hidden_size")
        lin_layer_size: int = params.get("linear_size")

        self.hidden_sizes: List[int] = [rnn_layer_size] * self.num_rnn_layers
        self.linear_sizes: List[int] = [lin_layer_size] * (self.num_lin_layers - 1)
        # ----------------------

        self.training_step_losses = []
        self.validation_step_losses = []
        self.training_step_accs = []
        self.validation_step_accs = []
        self.training_step_f1 = []
        self.validation_step_f1 = []

        # Create the RNN layers
        self.rnns = torch.nn.ModuleList([])
        self.rnns.append(torch.nn.RNNCell(2, self.hidden_sizes[0]))
        for layer in range(self.num_rnn_layers - 1):
            self.rnns.append(
                torch.nn.RNNCell(self.hidden_sizes[layer], self.hidden_sizes[layer + 1])
            )

        # Create the linear layers
        self.lins = torch.nn.ModuleList([])
        if self.num_lin_layers == 1:
            self.lins.append(torch.nn.Linear(self.hidden_sizes[-1], 2))
        elif self.num_lin_layers > 1:
            self.lins.append(
                torch.nn.Linear(self.hidden_sizes[-1], self.linear_sizes[0])
            )
            for layer in range(self.num_lin_layers - 2):
                self.lins.append(
                    torch.nn.Linear(
                        self.linear_sizes[layer], self.linear_sizes[layer + 1]
                    )
                )
            self.lins.append(torch.nn.Linear(self.linear_sizes[-1], 2))
        self.dropout = torch.nn.Dropout(p=dropout)

        # takes care of dtype
        self.to(torch.double)

    def _init_hidden(self, shape0: int, hidden_shapes: int) -> List[torch.Tensor]:
        return [
            torch.zeros(shape0, hidden_shape, dtype=torch.double).to(self.device)
            for hidden_shape in hidden_shapes
        ]

    def forward(self, input_t: torch.Tensor) -> torch.Tensor:
        # h_ts[i].shape = [features, hidden_size]
        h_ts = self._init_hidden(input_t.shape[0], self.hidden_sizes)

        for input in input_t.split(1, dim=2):
            input = input.squeeze(2)

            # rnn layers
            h_ts[0] = self.rnns[0](input, h_ts[0])
            h_ts[0] = self.dropout(h_ts[0])
            for i in range(1, self.num_rnn_layers):
                h_ts[i] = self.rnns[i](h_ts[i - 1], h_ts[i])
                h_ts[i] = self.dropout(h_ts[i])

            # linear layers
            output = self.lins[0](h_ts[-1])
            for i in range(1, self.num_lin_layers):
                output = self.lins[i](output)

        # just take the last output
        return output

    def configure_optimizers(self) -> optim.Optimizer:
        if self.optimizer == "adam":
            return optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)
        elif self.optimizer == "rmsprop":
            return optim.RMSprop(self.parameters(), lr=self.lr)
        elif self.optimizer == "sgd":
            return optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, nesterov=True)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        inputs: torch.Tensor
        inputs, targets = batch

        predicted = self(inputs)
        loss = torch.nn.functional.cross_entropy(predicted, targets)
        accuracy = torchmetrics.functional.accuracy(
            predicted.softmax(dim=1), targets, task="binary"
        )
        f1 = torchmetrics.functional.f1_score(
            predicted.softmax(dim=1), targets, task="binary"
        )

        self.log_dict(
            {"loss/train": loss, "acc/train": accuracy, "f1/train": f1},
            on_epoch=True,
            prog_bar=True,
            on_step=False,
            sync_dist=False,
        )
        self.training_step_losses.append(loss)
        self.training_step_accs.append(accuracy)
        self.training_step_f1.append(f1)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        inputs: torch.Tensor
        inputs, targets = batch

        predicted = self(inputs)
        loss = torch.nn.functional.cross_entropy(predicted, targets)
        accuracy = torchmetrics.functional.accuracy(
            predicted.softmax(dim=1), targets, task="binary"
        )
        f1 = torchmetrics.functional.f1_score(
            predicted.softmax(dim=1), targets, task="binary"
        )

        self.log_dict(
            {"loss/val": loss, "acc/val": accuracy, "f1/val": f1},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=False,
        )
        self.validation_step_losses.append(loss)
        self.validation_step_accs.append(accuracy)
        self.validation_step_f1.append(f1)
        return loss

    def predict_step(self, batch, batch_idx) -> dict:
        inputs: torch.Tensor
        inputs, targets = batch

        predicted = self(inputs[0])
        loss = torch.nn.functional.cross_entropy(predicted, targets[0])
        accuracy = torchmetrics.functional.accuracy(
            predicted.softmax(dim=1), targets[0], task="binary"
        )
        f1 = torchmetrics.functional.f1_score(
            predicted.softmax(dim=1), targets[0], task="binary"
        )

        predicted_labels = [
            1 - torch.round(pred_prob[0]) for pred_prob in predicted.softmax(dim=1)
        ]

        return {
            "loss": loss,
            "accuracy": accuracy,
            "f1": f1,
            "predicted_labels": predicted_labels,
        }


class Data(pl.LightningDataModule):
    def __init__(
        self,
        params: dict,
        binary: bool,
        plot_data: bool = False,
        print_split: bool = False,
        K: List[float] = None,
        train_size: float = 1.0,
        map_object=None,
        data_path=None,
        reduce_init_points=False,
    ):
        super(Data, self).__init__()
        if map_object is not None:
            map_object.generate_data(lyapunov=True)
            self.thetas, self.ps = map_object.retrieve_data()
            self.spectrum = map_object.retrieve_spectrum(binary=binary)
            if plot_data:
                map_object.plot_data()

        else:
            if not isinstance(K, list):
                K = [K]
            thetas, ps, self.spectrum = self._load_data(data_path, K)
            steps = params.get("steps")
            # too many steps in saved data
            self.thetas = thetas[:steps]
            self.ps = ps[:steps]
            if reduce_init_points:
                self.thetas = self.thetas[:, : params.get("init_points")]
                self.ps = self.ps[:, : params.get("init_points")]
                self.spectrum = self.spectrum[: params.get("init_points")]
            if binary:
                self.spectrum = (self.spectrum * 1e5 > 10).astype(int)
            if plot_data:
                self.plot_data(self.thetas, self.ps, self.spectrum)

        self.batch_size: int = params.get("batch_size")
        self.shuffle_paths: bool = params.get("shuffle_paths")
        self.shuffle_batches: bool = params.get("shuffle_batches")

        self.rng = np.random.default_rng(seed=42)

        # data.shape = [init_points, 2, steps]
        self.data = np.stack([self.thetas.T, self.ps.T], axis=1)

        # first shuffle trajectories and then make sequences
        if self.shuffle_paths:
            indices = np.arange(len(self.spectrum))
            self.rng.shuffle(indices)
            self.data = self.data[indices]
            self.spectrum = self.spectrum[indices]

        xy_pairs = self._make_input_output_pairs(self.data, self.spectrum)

        t = int(len(xy_pairs) * train_size)
        self.train_data = xy_pairs[:t]
        self.val_data = xy_pairs[t:]

        if print_split:
            print()
            print(f"Data shape: {self.data.shape}")
            print(
                f"Train data shape: {len(self.train_data)} pairs of shape ({len(self.train_data[0][0][0])}, {1})"
            )
            if train_size < 1.0:
                print(
                    f"Validation data shape: {len(self.val_data)} pairs of shape ({len(self.val_data[0][0][0])}, {1})"
                )
            print()

    def _make_input_output_pairs(self, data: np.ndarray, spectrum: list) -> list:
        return [
            (data[point], [1 - spectrum[point], spectrum[point]])
            for point in range(data.shape[0])
        ]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            Dataset(self.train_data),
            batch_size=self.batch_size,
            shuffle=self.shuffle_batches,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            Dataset(self.val_data),
            batch_size=2 * self.batch_size,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader:
        self.spectrum = [
            [1 - self.spectrum[point], self.spectrum[point]]
            for point in range(self.data.shape[0])
        ]
        return DataLoader(Dataset([(self.data, self.spectrum)]))

    def _load_data(
        self, path: str, K: List[float] | float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        directories: List[str] = self._get_subdirectories(path, K)

        thetas_list, ps_list, spectrum_list = [], [], []
        for directory in directories:
            temp_thetas = np.load(os.path.join(directory, "theta_values.npy"))
            temp_ps = np.load(os.path.join(directory, "p_values.npy"))
            temp_spectrum = np.load(os.path.join(directory, "spectrum.npy"))

            thetas_list.append(temp_thetas)
            ps_list.append(temp_ps)
            spectrum_list.append(temp_spectrum)

        thetas = np.concatenate(thetas_list, axis=1)
        ps = np.concatenate(ps_list, axis=1)
        spectrum = np.concatenate(spectrum_list)

        return thetas, ps, spectrum

    def _get_subdirectories(self, directory: str, K: List[float] | float) -> List[str]:
        subdirectories = []
        for d in os.listdir(directory):
            if os.path.isdir(os.path.join(directory, d)):
                if float(d) in K:
                    subdirectories.append(os.path.join(directory, d))

        subdirectories.sort()
        return subdirectories

    def plot_data(
        self, thetas: np.ndarray, ps: np.ndarray, spectrum: np.ndarray
    ) -> None:
        plt.figure(figsize=(7, 4))
        chaotic_indices = np.where(np.array(spectrum) == 1)[0]
        regular_indices = np.where(np.array(spectrum) == 0)[0]
        plt.plot(
            thetas[:, chaotic_indices],
            ps[:, chaotic_indices],
            "ro",
            markersize=0.5,
        )
        plt.plot(
            thetas[:, regular_indices],
            ps[:, regular_indices],
            "bo",
            markersize=0.5,
        )
        legend_handles = [
            plt.scatter([], [], color="red", marker=".", label="Chaotic"),
            plt.scatter([], [], color="blue", marker=".", label="Regular"),
        ]
        plt.legend(handles=legend_handles)
        plt.xlabel(r"$\theta$")
        plt.ylabel("p")
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.show()


class CustomCallback(pl.Callback):
    def __init__(self, print: bool) -> None:
        super(CustomCallback, self).__init__()
        self.print = print
        self.min_train_loss = np.inf
        self.min_val_loss = np.inf
        self.max_train_acc = 0
        self.max_val_acc = 0

    def on_train_start(self, trainer, pl_module):
        trainer.logger.log_hyperparams(
            pl_module.hparams,
            {
                "metrics/min_val_loss": np.inf,
                "metrics/min_train_loss": np.inf,
                "metrics/max_val_acc": 0,
                "metrics/max_train_acc": 0,
            },
        )

    def on_train_epoch_end(self, trainer, pl_module):
        mean_loss = torch.stack(pl_module.training_step_losses).mean()
        if mean_loss < self.min_train_loss:
            self.min_train_loss = mean_loss
            pl_module.log(
                "metrics/min_train_loss",
                mean_loss,
                sync_dist=False,
            )
        mean_acc = torch.stack(pl_module.training_step_accs).mean()
        if mean_acc > self.max_train_acc:
            self.max_train_acc = mean_acc
            pl_module.log(
                "metrics/max_train_acc",
                mean_acc,
                sync_dist=False,
            )
        pl_module.training_step_losses.clear()
        pl_module.training_step_accs.clear()

    def on_validation_epoch_end(self, trainer, pl_module):
        mean_loss = torch.stack(pl_module.validation_step_losses).mean()
        if mean_loss < self.min_val_loss:
            self.min_val_loss = mean_loss
            pl_module.log(
                "metrics/min_val_loss",
                mean_loss,
                sync_dist=False,
            )
        mean_acc = torch.stack(pl_module.validation_step_accs).mean()
        if mean_acc > self.max_val_acc:
            self.max_val_acc = mean_acc
            pl_module.log(
                "metrics/max_val_acc",
                mean_acc,
                sync_dist=False,
            )
        pl_module.validation_step_losses.clear()
        pl_module.validation_step_accs.clear()

    def on_fit_start(self, trainer, pl_module):
        if self.print:
            print()
            print("Training started!")
            print()
            self.t_start = time.time()

    def on_fit_end(self, trainer, pl_module):
        if self.print:
            print()
            print("Training ended!")
            train_time = time.time() - self.t_start
            print(f"Training time: {timedelta(seconds=train_time)}")
            print()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: np.ndarray):
        self.data: np.ndarray = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.data[idx]
        x = torch.tensor(x).to(torch.double)
        y = torch.tensor(y).to(torch.double)
        return x, y


class Gridsearch:
    def __init__(self, path: str, use_defaults: bool = False) -> None:
        self.path = path
        self.use_defaults = use_defaults

    def __next__(self):
        return self.update_params()

    def __iter__(self):
        for _ in range(10**3):
            yield self.update_params()

    def update_params(self) -> dict:
        params = read_yaml(self.path)
        if not self.use_defaults:
            params = self._update_params(params)

        try:
            del params["gridsearch"]
        except KeyError:
            pass

        return params

    def _update_params(self, params) -> dict:
        # don't use any seed
        rng: np.random.Generator = np.random.default_rng()
        for key, space in params.get("gridsearch").items():
            type = space.get("type")
            if type == "int":
                params[key] = int(rng.integers(space["lower"], space["upper"] + 1))
            elif type == "choice":
                list = space.get("list")
                choice = rng.choice(list)
                try:
                    choice = float(choice)
                except:
                    choice = str(choice)
                params[key] = choice
            elif type == "float":
                params[key] = rng.uniform(space["lower"], space["upper"])
            # print(f"{key} = {params[key]}")

        # to add variable layer size
        # if "layers" in key:
        #         num_layers = params[key]
        #         space = space["layer_sizes"]
        #         layer_type = space["layer_type"] + "_sizes"
        #         params[layer_type] = []
        #         for _ in range(num_layers):
        #             layer_size = rng.integers(space["lower"], space["upper"] + 1)
        #             params[layer_type].append(int(layer_size))
        #             if not space["varied"]:
        #                 params[layer_type][-1] = params[layer_type][0]
        #         if layer_type == "lin_sizes":
        #             params[layer_type] = params[layer_type][:-1]
        # print(f"{layer_type}: {params[layer_type]}")

        return params
