import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import os
import numpy as np
from typing import Tuple, List

from src.utils import plot_labeled_data


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
        self.shuffle_within_batches: bool = params.get("shuffle_within_batches")
        self.rng = np.random.default_rng(seed=42)

        # generate new data
        if map_object is not None:
            map_object.generate_data(lyapunov=True)
            self.thetas_orig, self.ps_orig = map_object.retrieve_data()
            self.spectrum = map_object.retrieve_spectrum(binary=binary)

        # load data
        elif data_path is not None:
            # NOTE: Training data: for each K, the shape of loaded data is (1000, 2601)
            # NOTE: Test data: for each K, the shape of loaded data is (1000, 200)
            self.thetas_orig, self.ps_orig, self.spectrum = self._load_data(
                data_path, K, binary
            )

            # NOTE: loaded data contains redundant steps
            steps = params.get("steps")
            self.thetas_orig = self.thetas_orig[:steps]
            self.ps_orig = self.ps_orig[:steps]

        # only take the first seq_len steps
        seq_len = params.get("seq_len")
        self.thetas = self.thetas_orig[:seq_len]
        self.ps = self.ps_orig[:seq_len]

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

        # data.shape = [init_points * len(K), seq_len, 2]
        self.data = np.stack([self.thetas.T, self.ps.T], axis=-1)

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
            shuffle=self.shuffle_within_batches,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            Dataset(self.input_output_pairs[self.t :]),
            batch_size=self.batch_size * 5,
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
        x = torch.tensor(x)
        y = torch.tensor(y)
        return x, y
