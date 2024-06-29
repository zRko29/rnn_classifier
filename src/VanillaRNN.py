import torch
from src.BaseRNN import BaseRNN


class Vanilla(BaseRNN):
    def __init__(self, **params):
        super(Vanilla, self).__init__(**params)

        # Create the RNN layers
        self.rnns = torch.nn.ModuleList([])
        self.rnns.append(
            torch.nn.RNNCell(2, self.hidden_sizes[0], nonlinearity=self.nonlin_hidden)
        )
        for layer in range(self.num_rnn_layers - 1):
            self.rnns.append(
                torch.nn.RNNCell(
                    self.hidden_sizes[layer],
                    self.hidden_sizes[layer + 1],
                    nonlinearity=self.nonlin_hidden,
                )
            )

        self.create_linear_layers()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dtype)
        x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()

        # h_ts[i].shape = [batch_size, hidden_sizes[i]]
        h_ts = self._init_hidden(batch_size, self.hidden_sizes)

        # rnn layers
        for t in range(seq_len):
            h_ts[0] = self.rnns[0](x[t], h_ts[0])
            for layer in range(1, self.num_rnn_layers):
                h_ts[layer] = self.rnns[layer](h_ts[layer - 1], h_ts[layer])

        output = h_ts[-1]

        # linear layers
        for layer in range(self.num_lin_layers):
            output = self.lins[layer](output)
            output = self.nonlin_lin(output)

        return output
