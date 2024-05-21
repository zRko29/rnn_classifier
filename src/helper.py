import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import torchmetrics

import os
import numpy as np
from typing import Tuple, List, Dict

from src.utils import plot_labeled_data


class Model(pl.LightningModule):
    def __init__(self, **params):
        super(Model, self).__init__()
        self.save_hyperparameters()

        self.num_rnn_layers: int = params.get("num_rnn_layers")
        self.num_lin_layers: int = params.get("num_lin_layers")
        dropout: float = params.get("dropout")
        self.lr: float = params.get("lr")
        self.optimizer: str = params.get("optimizer")

        self.non_lin = self.configure_non_linearity(params.get("non_linearity"))

        self.accuracy = torchmetrics.Accuracy(task="binary", threshold=0.5)
        self.f1 = torchmetrics.F1Score(task="binary", threshold=0.5)
        self.precision = torchmetrics.Precision(task="binary", threshold=0.5)
        self.recall = torchmetrics.Recall(task="binary", threshold=0.5)

        # ----------------------
        # NOTE: This logic is kept so that variable layer sizes can be reimplemented in the future
        rnn_layer_size: int = params.get("hidden_size")
        lin_layer_size: int = params.get("linear_size")

        self.hidden_sizes: List[int] = [rnn_layer_size] * self.num_rnn_layers
        self.linear_sizes: List[int] = [lin_layer_size] * (self.num_lin_layers - 1)
        # ----------------------

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

    def set_weight(self, labels: List[int]) -> torch.Tensor:
        pos_neg_ratio = (len(labels) - sum(labels)) / sum(labels)
        pos_weight = 2 * pos_neg_ratio / (1 + pos_neg_ratio)
        neg_weight = 2 - pos_weight
        self.weight = torch.tensor([neg_weight, pos_weight])

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
            output = self.non_lin(output)
            for i in range(1, self.num_lin_layers):
                output = self.lins[i](output)
                output = self.non_lin(output)

        # just take the last output
        return output

    def configure_non_linearity(self, non_linearity: str) -> torch.nn.Module:
        if non_linearity is None:
            return torch.nn.Identity()
        elif non_linearity.lower() == "relu":
            return torch.nn.ReLU()
        elif non_linearity.lower() == "leaky_relu":
            return torch.nn.LeakyReLU()
        elif non_linearity.lower() == "tanh":
            return torch.nn.Tanh()
        elif non_linearity.lower() == "elu":
            return torch.nn.ELU()
        elif non_linearity.lower() == "selu":
            return torch.nn.SELU()

    def configure_optimizers(self) -> optim.Optimizer:
        if self.optimizer == "adam":
            return optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)
        elif self.optimizer == "rmsprop":
            return optim.RMSprop(self.parameters(), lr=self.lr)
        elif self.optimizer == "sgd":
            return optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, nesterov=True)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        inputs: torch.Tensor
        targets: torch.Tensor
        inputs, targets = batch

        predicted = self(inputs)

        loss, accuracy, f1, precision, recall = self.compute_scores(predicted, targets)
        self.log_dict(
            {"loss/train": loss, "f1/train": f1},
            on_epoch=True,
            prog_bar=True,
            on_step=False,
        )
        self.log_dict(
            {"acc/train": accuracy, "prec/train": precision, "rec/train": recall},
            on_epoch=True,
            prog_bar=False,
            on_step=False,
        )
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        inputs: torch.Tensor
        targets: torch.Tensor
        inputs, targets = batch

        predicted = self(inputs)

        loss, accuracy, f1, precision, recall = self.compute_scores(predicted, targets)
        self.log_dict(
            {"loss/val": loss, "f1/val": f1},
            on_epoch=True,
            prog_bar=True,
            on_step=False,
        )
        self.log_dict(
            {"acc/val": accuracy, "prec/val": precision, "rec/val": recall},
            on_epoch=True,
            prog_bar=False,
            on_step=False,
        )
        return loss

    def predict_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        inputs: torch.Tensor
        targets: torch.Tensor
        inputs, targets = batch

        predicted = self(inputs[0])
        self.weight = torch.tensor([1, 1])

        loss, accuracy, f1, precision, recall = self.compute_scores(
            predicted, targets[0]
        )

        predicted_labels = self.invert_one_hot_labels(predicted.softmax(dim=1).round())

        return {
            "loss": loss,
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "predicted_labels": predicted_labels,
        }

    def compute_scores(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        loss = torch.nn.functional.cross_entropy(
            predictions,
            targets,
            weight=self.weight.to(predictions.device),
        )
        pred = self.invert_one_hot_labels(predictions)
        # use predictions = (predictions.softmax(dim=1)[:, 1] >= threshold).long() to support different thresholds

        targets = self.invert_one_hot_labels(targets)

        accuracy = self.accuracy(pred, targets)
        f1 = self.f1(pred, targets)
        precision = self.precision(pred, targets)
        recall = self.recall(pred, targets)

        return loss, accuracy, f1, precision, recall

    def invert_one_hot_labels(self, labels: torch.Tensor) -> torch.Tensor:
        # works because one-hot encoding is [max, min] = 0 and [min, max] = 1
        return torch.argmax(labels, axis=1)

    @rank_zero_only
    def on_train_start(self):
        """
        Required to add best_loss to hparams in logger.
        """
        self._trainer.logger.log_hyperparams(self.hparams, {"best_loss": np.inf})

    def on_train_epoch_end(self):
        """
        Required to log best_loss at the end of the epoch. sync_dist=True is required to average the best_loss over all devices.
        """
        best_loss = self._trainer.callbacks[-1].best_model_score or np.inf
        self.log("best_loss", best_loss, sync_dist=True)


class Data(pl.LightningDataModule):
    def __init__(
        self,
        params: dict,
        binary: bool,
        train_size: float,
        plot_data: bool = False,
        K: List[float] | float = None,
        map_object=None,
        data_path=None,
        reduce_init_points=False,
    ):
        super(Data, self).__init__()
        self.batch_size: int = params.get("batch_size")
        self.shuffle_trajectories: bool = params.get("shuffle_trajectories")
        self.shuffle_batches: bool = params.get("shuffle_batches")
        self.rng = np.random.default_rng(seed=42)

        self.thetas: np.ndarray
        self.ps: np.ndarray

        # generate new data
        if map_object is not None:
            map_object.generate_data(lyapunov=True)
            self.thetas, self.ps = map_object.retrieve_data()
            self.spectrum = map_object.retrieve_spectrum(binary=binary)

        # load data
        elif data_path is not None:
            # NOTE: Training data: for each K, the shape of loaded data is (1000, 2601)
            # NOTE: Test data: for each K, the shape of loaded data is (1000, 100)
            self.thetas, self.ps, self.spectrum = self._load_data(data_path, K, binary)

            # NOTE: loaded data contains redundant steps
            seq_len = params.get("seq_len")
            self.thetas = self.thetas[:seq_len]
            self.ps = self.ps[:seq_len]

        # shuffle trajectories
        # NOTE: This will shuffle trajectories between different K
        if self.shuffle_trajectories:
            indices = np.arange(len(self.spectrum))
            self.rng.shuffle(indices)
            self.thetas = self.thetas[:, indices]
            self.ps = self.ps[:, indices]
            self.spectrum = self.spectrum[indices]

            # NOTE: This will set the number of trajectories kept across all K, makes sense if data is shuffled
            if reduce_init_points and data_path is not None:
                max_points = params.get("init_points")
                self.thetas = self.thetas[:, :max_points]
                self.ps = self.ps[:, :max_points]
                self.spectrum = self.spectrum[:max_points]

        if plot_data:
            plot_labeled_data(self.thetas, self.ps, self.spectrum, "Labeled data")

        # data.shape = [init_points * len(K), 2, seq_len]
        self.data = np.stack([self.thetas.T, self.ps.T], axis=1)

        self.input_output_pairs = self._make_input_output_pairs(
            self.data, self.spectrum
        )
        self.t = int(len(self.input_output_pairs) * train_size)

    def _make_input_output_pairs(self, data: np.ndarray, spectrum: list) -> List:
        # (trajectory, label)
        one_hot = self.one_hot_labels(spectrum)
        return [(data[point], one_hot[point]) for point in range(data.shape[0])]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            Dataset(self.input_output_pairs[: self.t]),
            batch_size=self.batch_size,
            shuffle=self.shuffle_batches,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            Dataset(self.input_output_pairs[self.t :]),
            batch_size=2 * self.batch_size,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader:
        one_hot = self.one_hot_labels(self.spectrum)
        return DataLoader(Dataset([(self.data, one_hot)]))

    def one_hot_labels(self, labels: np.ndarray) -> np.ndarray:
        return [[1 - label, label] for label in labels]

    def _load_data(
        self, path: str, K: List[float] | float, binary: bool
    ) -> Tuple[np.ndarray]:
        if not isinstance(K, list):
            K = [K]

        directories: List[str] = self._get_subdirectories(path, K)

        thetas_list = [
            np.load(os.path.join(directory, "theta_values.npy"))
            for directory in directories
        ]
        ps_list = [
            np.load(os.path.join(directory, "p_values.npy"))
            for directory in directories
        ]
        spectrum_list = [
            np.load(os.path.join(directory, "spectrum.npy"))
            for directory in directories
        ]

        thetas = np.concatenate(thetas_list, axis=1)
        ps = np.concatenate(ps_list, axis=1)
        spectrum = np.concatenate(spectrum_list)

        if binary:
            spectrum = (spectrum * 1e5 > 11).astype(int)

        return thetas, ps, spectrum

    def _get_subdirectories(self, directory: str, K: List[float]) -> List[str]:
        subdirectories = []
        for d in os.listdir(directory):
            if os.path.isdir(os.path.join(directory, d)):
                if float(d) in K:
                    subdirectories.append(os.path.join(directory, d))

        subdirectories.sort()
        return subdirectories


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
