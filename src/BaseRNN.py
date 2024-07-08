import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import torchmetrics

from typing import Tuple, List, Dict


class BaseRNN(pl.LightningModule):
    def __init__(self, **params):
        super(BaseRNN, self).__init__()
        self.save_hyperparameters()

        self.num_rnn_layers: int = params.get("num_rnn_layers")
        self.num_lin_layers: int = params.get("num_lin_layers")
        self.dropout: float = params.get("dropout")
        self.lr: float = params.get("lr")
        self.optimizer: str = params.get("optimizer")

        self.nonlin_hidden = params.get("nonlinearity_hidden")
        self.nonlin_lin = self.configure_non_linearity(params.get("nonlinearity_lin"))
        self.accuracy, self.precision, self.recall, self.specificity = (
            self.configure_metrics(threshold=0.5)
        )

        # ------------------------------------------
        # NOTE: This logic is for variable layer sizes
        hidden_sizes: List[int] = params.get("hidden_sizes")
        linear_sizes: List[int] = params.get("linear_sizes")

        rnn_layer_size: int = params.get("hidden_size")
        lin_layer_size: int = params.get("linear_size")

        self.hidden_sizes: List[int] = (
            hidden_sizes or [rnn_layer_size] * self.num_rnn_layers
        )
        self.linear_sizes: List[int] = linear_sizes or [lin_layer_size] * (
            self.num_lin_layers - 1
        )

    def create_linear_layers(self):
        self.lins = nn.ModuleList([])

        if self.num_lin_layers == 1:
            self.lins.append(nn.Linear(self.hidden_sizes[-1], 2))
        elif self.num_lin_layers > 1:
            self.lins.append(nn.Linear(self.hidden_sizes[-1], self.linear_sizes[0]))
            for layer in range(self.num_lin_layers - 2):
                self.lins.append(
                    nn.Linear(self.linear_sizes[layer], self.linear_sizes[layer + 1])
                )
            self.lins.append(nn.Linear(self.linear_sizes[-1], 2))

    def _init_hidden(self, shape0: int, hidden_shapes: int) -> List[torch.Tensor]:
        return [
            torch.zeros(shape0, hidden_shape, device=self.device)
            for hidden_shape in hidden_shapes
        ]

    def set_weight(self, labels: List[int]) -> torch.Tensor:
        pos_neg_ratio = (len(labels) - sum(labels)) / sum(labels)
        pos_weight = 2 * pos_neg_ratio / (1 + pos_neg_ratio)
        neg_weight = 2 - pos_weight
        self.weight = torch.tensor([neg_weight, pos_weight])
        # self.weight = torch.tensor([1, 1])

    def configure_metrics(self, threshold: float) -> None:
        accuracy = torchmetrics.Accuracy(task="binary", threshold=threshold)
        precision = torchmetrics.Precision(task="binary", threshold=threshold)
        recall = torchmetrics.Recall(task="binary", threshold=threshold)
        specificity = torchmetrics.Specificity(task="binary", threshold=threshold)
        return accuracy, precision, recall, specificity

    def configure_non_linearity(self, non_linearity: str) -> nn.Module:
        if non_linearity == "relu":
            return F.relu
        elif non_linearity == "leaky_relu":
            return F.leaky_relu
        elif non_linearity == "tanh":
            return F.tanh
        elif non_linearity == "elu":
            return F.elu
        elif non_linearity == "selu":
            return F.selu

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if self.optimizer == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)
        elif self.optimizer == "rmsprop":
            return torch.optim.RMSprop(self.parameters(), lr=self.lr)
        elif self.optimizer == "sgd":
            return torch.optim.SGD(
                self.parameters(), lr=self.lr, momentum=0.9, nesterov=True
            )

    def training_step(self, batch, _) -> torch.Tensor:
        inputs: torch.Tensor
        targets: torch.Tensor
        inputs, targets = batch

        predicted = self(inputs)

        loss, accuracy, precision, recall, specificity, f1, balanced_accuracy = (
            self.compute_scores(predicted, targets)
        )
        self.log_dict(
            {
                "loss/train": loss,
                "f1/train": f1,
                "balanced_acc/train": balanced_accuracy,
            },
            on_epoch=True,
            prog_bar=True,
            on_step=False,
        )
        self.log_dict(
            {
                "acc/train": accuracy,
                "prec/train": precision,
                "rec/train": recall,
                "spec/train": specificity,
            },
            on_epoch=True,
            prog_bar=False,
            on_step=False,
        )
        return loss

    def validation_step(self, batch, _) -> torch.Tensor:
        inputs: torch.Tensor
        targets: torch.Tensor
        inputs, targets = batch

        predicted = self(inputs)

        loss, accuracy, precision, recall, specificity, f1, balanced_accuracy = (
            self.compute_scores(predicted, targets)
        )
        self.log_dict(
            {"loss/val": loss, "f1/val": f1, "balanced_acc/val": balanced_accuracy},
            on_epoch=True,
            prog_bar=True,
            on_step=False,
        )
        self.log_dict(
            {
                "acc/val": accuracy,
                "prec/val": precision,
                "rec/val": recall,
                "spec/val": specificity,
            },
            on_epoch=True,
            prog_bar=False,
            on_step=False,
        )
        return loss

    def predict_step(self, batch, _) -> Dict[str, torch.Tensor]:
        inputs: torch.Tensor
        targets: torch.Tensor
        inputs, targets = batch

        predicted = self(inputs[0])
        self.weight = torch.tensor([1, 1])

        loss, accuracy, precision, recall, specificity, f1, balanced_accuracy = (
            self.compute_scores(predicted, targets[0])
        )

        predicted_labels = self.invert_one_hot_labels(predicted.softmax(dim=1).round())

        return {
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "balanced_accuracy": balanced_accuracy,
            "predicted_labels": predicted_labels,
        }

    def compute_scores(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        loss = F.cross_entropy(
            predictions.to(self.dtype),
            targets.to(self.dtype),
            weight=self.weight.to(dtype=self.dtype, device=self.device),
        )
        pred = self.invert_one_hot_labels(predictions)
        # use predictions = (predictions.softmax(dim=1)[:, 1] >= threshold).long() to support different thresholds

        targets = self.invert_one_hot_labels(targets)

        accuracy = self.accuracy(pred, targets)
        precision = self.precision(pred, targets)
        recall = self.recall(pred, targets)
        specificity = self.specificity(pred, targets)

        f1 = 2 * (precision * recall) / max(precision + recall, 1)
        balanced_accuracy = (recall + specificity) / 2

        return loss, accuracy, precision, recall, specificity, f1, balanced_accuracy

    def invert_one_hot_labels(self, labels: torch.Tensor) -> torch.Tensor:
        # works because one-hot encoding is [max, min] = 0 and [min, max] = 1
        return torch.argmax(labels, axis=1)

    @rank_zero_only
    def on_train_start(self):
        """
        Required to add best_loss to hparams in logger. Used for gridsearch.
        """
        self._trainer.logger.log_hyperparams(self.hparams, {"best_acc": 0})
        self._trainer.logger.log_hyperparams(self.hparams, {"best_loss": float("inf")})

    def on_train_epoch_end(self):
        """
        Required to log best_loss at the end of the epoch. sync_dist=True is required to average the best_loss over all devices.
        """
        best_score = self._trainer.callbacks[-1].best_model_score or 0
        self.log("best_acc", best_score, sync_dist=True)
        # best_score = self._trainer.callbacks[-1].best_model_score or float("inf")
        # self.log("best_loss", best_score, sync_dist=True)


class ResidualRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ResidualRNNCell, self).__init__()

        self.weight_1 = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_2 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight_3 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_1)
        nn.init.kaiming_uniform_(self.weight_2)
        nn.init.kaiming_uniform_(self.weight_3)
        nn.init.zeros_(self.bias)

    def forward(self, input1, input2, delayed_input=None):
        output = F.linear(input1, self.weight_1, self.bias)
        output += F.linear(input2, self.weight_2)

        if delayed_input is not None:
            output += F.linear(delayed_input, self.weight_3)

        output = F.tanh(output)

        return output


class MinimalGatedCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MinimalGatedCell, self).__init__()

        # Parameters for forget gate
        self.weight_fx = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_fh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_f = nn.Parameter(torch.Tensor(hidden_size))

        # Parameters for candidate activation
        self.weight_hx = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_h = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_fx)
        nn.init.kaiming_uniform_(self.weight_fh)
        nn.init.zeros_(self.bias_f)

        nn.init.kaiming_uniform_(self.weight_hx)
        nn.init.kaiming_uniform_(self.weight_hf)
        nn.init.zeros_(self.bias_h)

    def forward(self, input1, input2):
        # Compute forget gate
        f_t = F.linear(input1, self.weight_fx, self.bias_f)
        f_t += F.linear(input2, self.weight_fh)
        f_t = F.sigmoid(f_t)

        # Compute candidate activation
        h_hat_t = F.linear(input1, self.weight_hx, self.bias_h)
        h_hat_t += F.linear(f_t * input2, self.weight_hf)
        h_hat_t = F.tanh(h_hat_t)

        # Compute output
        h_t = (1 - f_t) * input2 + f_t * h_hat_t

        return h_t
