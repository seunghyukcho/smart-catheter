import numpy as np
from torch.utils import data


def add_dataset_args(parser):
    group = parser.add_argument_group('dataset')
    group.add_argument('--input_len', type=int, default=100,
                       help="Lenght of input signal.")
    group.add_argument('--n_channel', type=int, default=3,
                       help="Number of channels in input signal.")


class Dataset(data.Dataset):
    def __init__(self, args, root_dir):
        self.input_len = args.input_len
        self.n_channel = args.n_channel

        self.signals = np.load(f"{root_dir}/signals.npz")["arr_0"]
        self.scales = np.load(f"{root_dir}/scales.npz")["arr_0"]

        self.signals = self.signals.reshape((self.signals.shape[0], -1, self.input_len * self.n_channel))

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.scales[idx]
