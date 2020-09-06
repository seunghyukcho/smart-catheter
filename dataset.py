import itertools
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class CatheterDataset(Dataset):
    def __init__(self, data_path):
        files = [str(x) for x in Path(data_path).glob("**/*.csv")]
        self.signals = []
        self.scales = []

        mn = 100000
        for file in files:
            x, y = [], []
            with open(file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    arr = line.strip().split(',')
                    arr = [float(val) for val in arr]
                    x.append(arr[:-1])
                    y.append(arr[-1])

            self.signals.append(x)
            self.scales.append(y)
            mn = min(mn, len(x))

        for idx in range(len(self.signals)):
            self.signals[idx] = self.signals[idx][:mn]
            self.scales[idx] = self.scales[idx][:mn]

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.scales[idx]


class RNNDataset(CatheterDataset):
    def __init__(self, data_path):
        super().__init__(data_path)

        self.signals = np.array(self.signals, dtype=np.float32)
        self.scales = np.array(self.scales, dtype=np.float32)


class FCNDataset(CatheterDataset):
    def __init__(self, data_path):
        super().__init__(data_path)

        self.signals = list(itertools.chain(*self.signals))
        self.scales = list(itertools.chain(*self.scales))

        self.signals = np.array(self.signals, dtype=np.float32)
        self.scales = np.array(self.scales, dtype=np.float32)
