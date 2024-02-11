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
from tqdm import tqdm


class Model(pl.LightningModule):
    def __init__(
        self,
        **params: dict,
    ):
        super(Model, self).__init__()
        self.save_hyperparameters()

        self.hidden_sizes = params.get("rnn_sizes")
        self.linear_sizes = params.get("lin_sizes")
        self.num_rnn_layers = params.get("num_rnn_layers")
        self.num_lin_layers = params.get("num_lin_layers")
        dropout = params.get("dropout")
        self.lr = params.get("lr")
        self.optimizer = params.get("optimizer")

        self.training_step_losses = []
        self.validation_step_losses = []
        self.training_step_accs = []
        self.validation_step_accs = []

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

    def _init_hidden(self, shape0: int, hidden_shapes: int):
        return [
            torch.zeros(shape0, hidden_shape, dtype=torch.double).to(self.device)
            for hidden_shape in hidden_shapes
        ]

    def forward(self, input_t):
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
            output = torch.relu(self.lins[0](h_ts[-1]))
            for i in range(1, self.num_lin_layers - 1):
                output = torch.relu(self.lins[i](output))
            output = self.lins[-1](output)

        # just take the last output
        return output

    def configure_optimizers(self):
        if self.optimizer == "adam":
            return optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)
        elif self.optimizer == "rmsprop":
            return optim.RMSprop(self.parameters(), lr=self.lr)
        elif self.optimizer == "sgd":
            return optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, nesterov=True)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        predicted = self(inputs)
        loss = torch.nn.functional.cross_entropy(predicted, targets)
        accuracy = torchmetrics.functional.accuracy(
            predicted.softmax(dim=1), targets, task="binary"
        )

        self.log_dict(
            {"loss/train": loss, "acc/train": accuracy},
            on_epoch=True,
            prog_bar=True,
            on_step=False,
        )
        self.training_step_losses.append(loss)
        self.training_step_accs.append(accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        predicted = self(inputs)
        loss = torch.nn.functional.cross_entropy(predicted, targets)
        accuracy = torchmetrics.functional.accuracy(
            predicted.softmax(dim=1), targets, task="binary"
        )

        self.log_dict(
            {"loss/val": loss, "acc/val": accuracy},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.validation_step_losses.append(loss)
        self.validation_step_accs.append(accuracy)
        return loss

    def predict_step(self, batch, batch_idx):
        inputs, targets = batch
        predicted = self(inputs[0])
        loss = torch.nn.functional.cross_entropy(predicted, targets[0])
        accuracy = torchmetrics.functional.accuracy(
            predicted.softmax(dim=1), targets[0], task="binary"
        )
        return {"loss": loss, "accuracy": accuracy}


class Data(pl.LightningDataModule):
    def __init__(
        self,
        print_split: bool,
        plot_data: bool,
        params: dict,
        binary: bool,
        K_upper_lim: float = None,
        train_size: float = 1.0,
        map_object=None,
        data_path=None,
    ):
        super(Data, self).__init__()
        if map_object is not None:
            map_object.generate_data(lyapunov=True)
            thetas, ps = map_object.retrieve_data()
            self.spectrum = map_object.retrieve_spectrum(binary=binary)
            if plot_data:
                map_object.plot_data()

        else:
            thetas, ps, self.spectrum = self._load_data(data_path, K_upper_lim)
            steps = params.get("steps")
            # too many steps in saved data
            thetas = thetas[:steps]
            ps = ps[:steps]
            if binary:
                self.spectrum = (self.spectrum > 1e-4).astype(int)
            if plot_data:
                self.plot_data(thetas, ps, self.spectrum)

        self.batch_size = params.get("batch_size")
        self.shuffle_paths = params.get("shuffle_paths")
        self.shuffle_batches = params.get("shuffle_batches")

        self.rng = np.random.default_rng(seed=42)

        # data.shape = [init_points, 2, steps]
        self.data = np.stack([thetas.T, ps.T], axis=1)

        # first shuffle trajectories and then make sequences
        if self.shuffle_paths:
            self.rng.shuffle(self.data)

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

    def _make_input_output_pairs(self, data, spectrum):
        return [
            (data[point], [1 - spectrum[point], spectrum[point]])
            for point in range(data.shape[0])
        ]

    def train_dataloader(self):
        return DataLoader(
            Dataset(self.train_data),
            batch_size=self.batch_size,
            shuffle=self.shuffle_batches,
        )

    def val_dataloader(self):
        return DataLoader(
            Dataset(self.val_data),
            batch_size=2 * self.batch_size,
            shuffle=False,
        )

    def predict_dataloader(self):
        self.spectrum = [
            [1 - self.spectrum[point], self.spectrum[point]]
            for point in range(self.data.shape[0])
        ]
        return DataLoader(Dataset([(self.data, self.spectrum)]))

    def _load_data(self, path, K_upper_lim):
        directories = self._get_subdirectories(path, K_upper_lim)
        bar_loader = tqdm(
            directories, desc="Loading data", unit=" directories", ncols=80
        )

        thetas_list, ps_list, spectrum_list = [], [], []
        for directory in bar_loader:
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

    def _get_subdirectories(self, directory, K_upper_lim):
        subdirectories = [
            os.path.join(directory, d)
            for d in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, d)) and float(d) <= K_upper_lim
        ]
        subdirectories.sort()
        return subdirectories

    def plot_data(self, thetas, ps, spectrum):
        plt.figure(figsize=(7, 4))
        chaotic_indices = np.where(spectrum == 1)[0]
        regular_indices = np.where(spectrum == 0)[0]
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
    def __init__(self):
        super(CustomCallback, self).__init__()
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
            pl_module.log("metrics/min_train_loss", mean_loss)
        mean_acc = torch.stack(pl_module.training_step_accs).mean()
        if mean_acc > self.max_train_acc:
            self.max_train_acc = mean_acc
            pl_module.log("metrics/max_train_acc", mean_acc)
        pl_module.training_step_losses.clear()
        pl_module.training_step_accs.clear()

    def on_validation_epoch_end(self, trainer, pl_module):
        mean_loss = torch.stack(pl_module.validation_step_losses).mean()
        if mean_loss < self.min_val_loss:
            self.min_val_loss = mean_loss
            pl_module.log("metrics/min_val_loss", mean_loss)
        mean_acc = torch.stack(pl_module.validation_step_accs).mean()
        if mean_acc > self.max_val_acc:
            self.max_val_acc = mean_acc
            pl_module.log("metrics/max_val_acc", mean_acc)
        pl_module.validation_step_losses.clear()
        pl_module.validation_step_accs.clear()

    def on_fit_start(self, trainer, pl_module):
        print()
        print("Training started!")
        print()
        self.t_start = time.time()

    def on_fit_end(self, trainer, pl_module):
        print()
        print("Training ended!")
        train_time = time.time() - self.t_start
        print(f"Training time: {timedelta(seconds=train_time)}")
        print()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        x = torch.tensor(x).to(torch.double)
        y = torch.tensor(y).to(torch.double)
        return x, y


class Gridsearch:
    def __init__(self, path, num_vertices):
        self.path = path
        self.grid_step = 1
        self.num_vertices = num_vertices

    def get_params(self):
        with open(os.path.join(self.path, "parameters.yaml"), "r") as file:
            params = yaml.safe_load(file)
            if self.num_vertices > 0:
                params = self._update_params(params)
                self.grid_step += 1
            if params.get("gridsearch") is not None:
                del params["gridsearch"]

        return params

    def _update_params(self, params):
        rng = np.random.default_rng()
        for key, space in params.get("gridsearch").items():
            dtype = space.get("dtype")
            if dtype == "int":
                params[key] = int(rng.integers(space["lower"], space["upper"] + 1))
            elif dtype == "bool":
                params[key] = rng.choice([True, False])
            elif dtype == "float":
                params[key] = rng.uniform(space["lower"], space["upper"])
            print(f"{key} = {params[key]}")

            if "layers" in key:
                num_layers = params[key]
                space = space["layer_sizes"]
                layer_type = space["layer_type"] + "_sizes"
                params[layer_type] = []
                for _ in range(num_layers):
                    layer_size = rng.integers(space["lower"], space["upper"] + 1)
                    params[layer_type].append(int(layer_size))
                    if not space["varied"]:
                        params[layer_type][-1] = params[layer_type][0]
                if layer_type == "lin_sizes":
                    params[layer_type] = params[layer_type][:-1]
                print(f"{layer_type}: {params[layer_type]}")

        print("-" * 80)
        print(f"Gridsearch step: {self.grid_step} / {self.num_vertices}")
        print()

        return params
