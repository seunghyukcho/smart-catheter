from torch import nn


def add_model_args(parser):
    group = parser.add_argument_group('model')
    group.add_argument('--n_hid', type=int, default=128,
                       help="Number of units in a FC layer.")
    group.add_argument('--n_layer', type=int, default=8,
                       help="Number of FC layers.")


class Model(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.input_dim = args.input_len * args.n_channel
        self.n_hid = args.n_hid
        self.n_layer = args.n_layer

        self.rnn = nn.GRU(
            self.input_dim,
            self.n_hid,
            self.n_layer,
            batch_first=True
        )
        self.decoder = nn.Linear(self.n_hid, 1)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)
        x = self.decoder(x)
        x = self.activation(x)

        return x