import numpy as np
import datetime
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def read_data(path, rate):
    data = pd.read_csv(path, header=None, index_col=1)
    data = data.drop(data.columns[0], axis=1)
    data = data.dropna()
    data.index = pd.to_datetime(data.index)
    data = data.resample(rate).mean()
    data = data.interpolate(method='cubic')

    return data


path = Path('data/raw/Test')
signal_files = list(path.glob('**/Itrrg*'))
scale_files = list(path.glob('**/Scale*'))

signal_files.sort()
scale_files.sort()

signal_out = []
scale_out = []
for idx in tqdm(range(len(signal_files))):
    signal_data = read_data(signal_files[idx], 'L')
    scale_data = read_data(scale_files[idx], '10L')

    signal_base = signal_data.iloc[0].name
    signal_base_idx = 0
    scale_base_idx = 0
    while True:
        scale_time = scale_data.iloc[scale_base_idx].name
        if scale_time - datetime.timedelta(milliseconds=99) < signal_base:
            scale_base_idx += 1
        else:
            break

    scale_base = scale_data.iloc[scale_base_idx].name
    while True:
        signal_time = signal_data.iloc[signal_base_idx].name
        if signal_time < scale_base - datetime.timedelta(milliseconds=99):
            signal_base_idx += 1
        else:
            break

    signals = []
    scales = []
    for i in range(scale_base_idx, len(scale_data), 10):
        start_idx = signal_base_idx + (i - scale_base_idx) * 10
        end_idx = start_idx + 100

        if end_idx > len(signal_data):
            break
        signals.append(signal_data.iloc[start_idx:end_idx].to_numpy())
        scales.append(scale_data.iloc[i].to_numpy())

    if len(signals) != 599:
        continue

    signals = np.stack(signals, axis=0)
    scales = np.stack(scales, axis=0)

    signal_out.append(signals)
    scale_out.append(scales)

signal_out = np.stack(signal_out, axis=0)
scale_out = np.stack(scale_out, axis=0)
print(signal_out.shape, scale_out.shape)

np.save('data/preprocess/test/signals.npy', signal_out)
np.save('data/preprocess/test/scales.npy', scale_out)
