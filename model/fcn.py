from torch import nn

from .base import BaseModel


class FCNModel(BaseModel):

    def __init__(self, args):
        super().__init__(args)

        fcn = self.stack_linear_layers(self.args.nhids, self.args.nlayers)
        self.encoder = nn.Linear(self.args.input_dim, self.args.nhids)
        self.fcn = nn.Sequential(*fcn)
        self.decoder = nn.Linear(self.args.nhids, 1)
        self.activation = nn.LeakyReLU()

    @staticmethod
    def stack_linear_layers(nhids, nlayers):
        return [nn.Sequential(
            nn.Linear(nhids, nhids),
            nn.BatchNorm1d(nhids),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        ) for _ in range(nlayers)]

    def forward(self, x):
        x = self.encoder(x)
        x = self.fcn(x)
        x = self.decoder(x)
        x = self.activation(x)

        return x
