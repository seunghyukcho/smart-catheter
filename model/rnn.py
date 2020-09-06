from torch import nn

from .base import BaseModel


class RNNModel(BaseModel):

    def __init__(self, options):
        super().__init__(options)
        self.rnn = nn.GRU(
            self.options.input_length * 3,
            self.options.nhids,
            self.options.nlayers,
            batch_first=True
        )
        self.decoder = nn.Linear(self.options.nhids, 1)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)
        x = self.decoder(x)
        x = self.activation(x)

        return x
