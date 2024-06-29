import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import ModelSummary


class Model(pl.LightningModule):
    def __init__(self):
        super(Model, self).__init__()

        self.example_input_array: torch.Tensor = torch.randn(256, 20, 2)

        # Create the RNN layers
        self.rnn1 = torch.nn.RNNCell(2, 200, nonlinearity="tanh")
        self.rnn2 = torch.nn.RNNCell(200, 200, nonlinearity="tanh")
        self.rnn3 = torch.nn.RNNCell(200, 200, nonlinearity="tanh")

        # Create the linear layers
        self.lin1 = torch.nn.Linear(200, 100)
        self.lin2 = torch.nn.Linear(100, 32)
        self.lin3 = torch.nn.Linear(32, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dtype)
        x = x.transpose(0, 1)

        h_ts1 = torch.zeros(256, 200)
        h_ts2 = torch.zeros(256, 200)
        h_ts3 = torch.zeros(256, 200)

        # rnn layers
        for t in range(20):
            h_ts1 = self.rnn1(x[t], h_ts1)
            h_ts2 = self.rnn2(h_ts1, h_ts2)
            h_ts3 = self.rnn3(h_ts3, h_ts3)

        outputs = h_ts3

        # linear layers
        outputs = self.lin1(outputs)
        outputs = torch.tanh(outputs)
        outputs = self.lin2(outputs)
        outputs = torch.tanh(outputs)
        outputs = self.lin3(outputs)

        return outputs


class HybridRNNCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, nonlinearity="relu"):
        super(HybridRNNCell, self).__init__()

        self.weight_ih = torch.nn.Parameter(torch.Tensor(hidden_size1, input_size))
        self.bias_ih = torch.nn.Parameter(torch.Tensor(hidden_size1))

        # hidden state at time t-1
        self.weight_hh1 = torch.nn.Parameter(torch.Tensor(hidden_size1, hidden_size1))
        self.bias_hh1 = torch.nn.Parameter(torch.Tensor(hidden_size1))

        # hidden state at previous layer
        self.weight_hh2 = torch.nn.Parameter(torch.Tensor(hidden_size1, hidden_size2))
        self.bias_hh2 = torch.nn.Parameter(torch.Tensor(hidden_size1))

        if nonlinearity == "tanh":
            self.nonlinearity = torch.tanh
        elif nonlinearity == "relu":
            self.nonlinearity = torch.nn.functional.relu

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight_ih)
        torch.nn.init.zeros_(self.bias_ih)

        torch.nn.init.kaiming_uniform_(self.weight_hh1)
        torch.nn.init.zeros_(self.bias_hh1)

        torch.nn.init.kaiming_uniform_(self.weight_hh2)
        torch.nn.init.zeros_(self.bias_hh2)

    def forward(self, input, hidden1, hidden2):
        h_t = self.nonlinearity(
            torch.nn.functional.linear(input, self.weight_ih, self.bias_ih)
            + torch.nn.functional.linear(hidden1, self.weight_hh1, self.bias_hh1)
            + torch.nn.functional.linear(hidden2, self.weight_hh2, self.bias_hh2)
        )
        return h_t


if __name__ == "__main__":
    model = Model()
    summary = ModelSummary(model, max_depth=-1)
    print(summary)
