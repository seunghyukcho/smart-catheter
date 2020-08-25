import itertools
from pathlib import Path
from torch.utils.data import Dataset


class CatheterDataset(Dataset):
    def __init__(self, data_path):
        files = [str(x) for x in Path(data_path).glob("**/*.csv")]
        self.signals = []
        self.scales = []

        for file in files:
            x, y = [], []
            with open(file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    arr = line.strip().split(',')
                    y.append(arr[1])
                    x.append(arr[0].split(' '))

            self.signals.append(x)
            self.scales.append(y)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.scales[idx]


class RNNDataset(CatheterDataset):
    pass


class FCNDataset(CatheterDataset):
    def __init__(self, data_path):
        super().__init__(data_path)

        self.signals = list(itertools.chain(*self.signals))
        self.scales = list(itertools.chain(*self.scales))