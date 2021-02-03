import torch
from torch import nn


def add_model_args(parser):
    group = parser.add_argument_group('model')
    group.add_argument('--n_hid', type=int, default=128,
                       help="Number of units in transformer.")
    group.add_argument('--n_layer', type=int, default=8,
                       help="Number of transformer layers.")
    group.add_argument('--n_head', type=int, default=12,
                       help="Number of heads in transformer.")


class Model(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.input_dim = args.input_len * args.n_channel
        self.n_head = args.n_head
        self.n_hid = args.n_hid
        self.n_layer = args.n_layer

        encoder_layer = nn.TransformerDecoderLayer(
            d_model=self.input_dim,
            nhead=self.n_head,
            dim_feedforward=self.n_hid
        )
        self.encoder = nn.TransformerDecoder(encoder_layer, num_layers=self.n_layer)
        self.decoder = nn.Linear(self.input_dim, 1)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        memory = torch.zeros(x.shape)
        x = self.encoder(x, memory)
        x = self.decoder(x)
        x = self.activation(x)

        return x
