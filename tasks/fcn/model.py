from torch import nn


def add_model_args(parser):
    group = parser.add_argument_group('model')
    group.add_argument('--n_hid', type=int, default=128,
                       help="Number of units in a FC layer.")
    group.add_argument('--n_layer', type=int, default=8,
                       help="Number of FC layers.")
    group.add_argument('--dropout', type=float, default=0.5,
                       help="Dropout rate.")


class Model(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.n_hid = args.n_hid
        self.n_layer = args.n_layer
        self.input_dim = args.input_len * args.n_channel
        self.dropout = args.dropout

        fcn = self.stack_linear_layers(self.n_hid, self.n_layer)
        self.encoder = nn.Linear(self.input_dim, self.n_hid)
        self.fcn = nn.Sequential(*fcn)
        self.decoder = nn.Linear(self.n_hid, 1)
        self.activation = nn.LeakyReLU()

    def stack_linear_layers(self, n_hid, n_layer):
        return [nn.Sequential(
            nn.Linear(n_hid, n_hid),
            nn.BatchNorm1d(n_hid),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout)
        ) for _ in range(n_layer)]

    def forward(self, x):
        x = self.encoder(x)
        x = self.fcn(x)
        x = self.decoder(x)
        x = self.activation(x)

        return x