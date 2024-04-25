import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional


class StandardMap:
    """
    A class representing the Standard Map dynamical system.
    """

    def __init__(
        self,
        init_points: int = None,
        steps: int = None,
        K: float = None,
        sampling: str = None,
        seed: bool = None,
        lyapunov_steps: int = 10**5,
        params: dict = None,
    ) -> None:
        self.init_points: int = init_points or params.get("init_points")
        self.steps: int = steps or params.get("steps")
        self.K: List[float] | float = K or params.get("K")
        self.sampling: str = sampling or params.get("sampling")

        self.rng: np.random.Generator = np.random.default_rng(seed=seed)
        self.lyapunov_steps: int = lyapunov_steps
        self.spectrum = np.array([])

    def retrieve_data(self) -> Tuple[np.ndarray]:
        return self.theta_values, self.p_values

    def retrieve_spectrum(
        self,
        binary: bool = False,
        threshold: int = 11,
    ) -> np.ndarray:
        if binary:
            self.spectrum = (self.spectrum * 1e5 > threshold).astype(int)
        return self.spectrum

    def save_data(self, data_path: str) -> None:
        np.save(f"{data_path}/theta_values.npy", self.theta_values)
        np.save(f"{data_path}/p_values.npy", self.p_values)
        np.save(f"{data_path}/spectrum.npy", self.spectrum)

    def generate_data(self, lyapunov: bool = False) -> None:
        steps: int = self.lyapunov_steps if lyapunov else self.steps

        # NOTE: returns data for one K, will be reused for all K
        theta_i: np.ndarray
        p_i: np.ndarray
        theta_i, p_i = self._get_initial_points()

        if not isinstance(self.K, list):
            K_list: List[float] = [self.K]
        else:
            if len(self.K) == 3 and isinstance(self.K[2], int):
                K_list: List[float] = np.linspace(*self.K)
            else:
                K_list: List[float] = self.K

        # shape: (steps, init_points * len(K_list))
        self.theta_values: np.ndarray = np.empty(
            (steps, self.init_points * len(K_list))
        )
        self.p_values: np.ndarray = np.empty((steps, self.init_points * len(K_list)))

        for i, K in enumerate(K_list):
            theta = theta_i.copy()
            p = p_i.copy()
            for step in range(steps):
                theta = np.mod(theta + p, 1)
                p = np.mod(p + K / (2 * np.pi) * np.sin(2 * np.pi * theta), 1)
                self.theta_values[
                    step, i * self.init_points : (i + 1) * self.init_points
                ] = theta
                self.p_values[
                    step, i * self.init_points : (i + 1) * self.init_points
                ] = p

        if lyapunov:
            self.spectrum = self._lyapunov(K_list)

        # reduce generated data
        self.theta_values = self.theta_values[: self.steps]
        self.p_values = self.p_values[: self.steps]
        self.lyapunov = lyapunov

    def _jaccobi(self, row: int, column: int, K: float) -> np.ndarray:
        der = K * np.cos(
            2 * np.pi * (self.theta_values[row, column] + self.p_values[row, column])
        )
        return np.array([[1, 1], [der, 1 + der]])

    def _lyapunov(self, K_list: List[float], treshold: int = 1e3) -> np.ndarray:
        spectrum = np.empty(self.theta_values.shape[1])
        for column in range(self.theta_values.shape[1]):
            M = np.identity(2)
            exp = np.zeros(2)

            for row in range(self.lyapunov_steps):
                M = (
                    self._jaccobi(
                        row,
                        column,
                        K_list[column // self.init_points],
                    )
                    @ M
                )

                if np.linalg.norm(M) > treshold:
                    Q, R = np.linalg.qr(M)
                    exp += np.log(np.abs(R.diagonal()))
                    M = Q

            _, R = np.linalg.qr(M)
            exp += np.log(np.abs(R.diagonal()))

            spectrum[column] = exp[0] / self.lyapunov_steps

        return spectrum

    def _get_initial_points(self) -> Tuple[np.ndarray, np.ndarray]:
        params: List = [0.01, 0.99, self.init_points]

        # sample randomly
        if self.sampling == "random":
            theta_init = self.rng.uniform(*params)
            p_init = self.rng.uniform(*params)

        # sample on diagonal
        elif self.sampling == "linear":
            theta_init = np.linspace(*params)
            p_init = np.linspace(*params)

        # sample on grid
        elif self.sampling == "grid":
            params = [0.01, 0.99, int(np.sqrt(self.init_points))]
            theta_init, p_init = np.meshgrid(np.linspace(*params), np.linspace(*params))
            theta_init = theta_init.flatten()
            p_init = p_init.flatten()

        return theta_init, p_init

    def plot_data(
        self,
        show_plot: bool = True,
        save_path: Optional[str] = None,
        threshold: int = 10,
    ) -> None:
        plt.figure(figsize=(7, 4))
        if self.lyapunov:
            spectrum = self.retrieve_spectrum(binary=True, threshold=threshold)
            chaotic_indices: np.ndarray[int] = np.where(spectrum == 1)[0]
            regular_indices: np.ndarray[int] = np.where(spectrum == 0)[0]
            plt.plot(
                self.theta_values[:, chaotic_indices],
                self.p_values[:, chaotic_indices],
                "ro",
                markersize=0.5,
            )
            plt.plot(
                self.theta_values[:, regular_indices],
                self.p_values[:, regular_indices],
                "bo",
                markersize=0.5,
            )
            legend_handles = [
                plt.scatter([], [], color="red", marker=".", label="Chaotic"),
                plt.scatter([], [], color="blue", marker=".", label="Regular"),
            ]
            plt.legend(handles=legend_handles)
        else:
            plt.plot(self.theta_values, self.p_values, "bo", markersize=0.5)
        plt.xlabel(r"$\theta$")
        plt.ylabel("p")
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        if save_path is not None:
            plt.savefig(save_path)
        if show_plot:
            plt.show()
        else:
            plt.close()


if __name__ == "__main__":
    # standard
    map = StandardMap(init_points=50, steps=200, sampling="random", K=0.1, seed=42)
    map.generate_data(lyapunov=True)
    spectrum = map.retrieve_spectrum()
    print(spectrum.shape)

    # vary chaos threshold
    # for K in [1.0, 1.5]:
    #     for threshold in np.arange(11.8, 13, 0.2):
    #         map = StandardMap(
    #             init_points=60, steps=500, sampling="random", K=K, seed=42
    #         )
    #         map.generate_data(lyapunov=True)
    #         map.plot_data(
    #             show_plot=False,
    #             threshold=threshold,
    #             save_path=f"plots/K_{K}/threshold_{round(threshold,1)}.pdf",
    #         )

    # save data
    # for K in np.arange(0.1, 2.1, 0.1):
    #     K = round(K, 1)
    #     path = "testing_data"
    #     if str(K) not in os.listdir(path):
    #         os.mkdir(f"{path}/{K}")
    #         map = StandardMap(
    #             init_points=100, steps=1000, sampling="random", K=K, seed=42
    #         )
    #         map.generate_data(lyapunov=True)
    #         map.save_data(data_path=f"{path}/{K}")
