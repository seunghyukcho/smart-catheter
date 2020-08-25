from torch import nn

from .base import BaseModel


class RNNModel(BaseModel):

    def __init__(self, args):
        super().__init__(args)

        self.rnn = nn.GRU(
            self.args.input_dim,
            self.args.nhids,
            self.args.nlayers,
            batch_first=True
        )
        self.decoder = nn.Linear(self.args.nhids, 1)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)
        x = self.decoder(x)
        x = self.activation(x)

        return x
