import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.nn.functional import l1_loss

from models import DNN, RNN, CNN
from data import import_data
from preprocess import ndata, frequency


def repackage_hidden(h):
    if h is None:
        return h
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def main():
    parser = argparse.ArgumentParser(description='Smart Catheter Predictor')
    parser.add_argument('--file-name', type=str, default='checkpoints/test/checkpoint_final.pth')
    parser.add_argument('--result-dir', type=str, default='results/test')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    args = parser.parse_args()

    torch.manual_seed(1)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f'Device selected: {device}\n')

    model = torch.load(args.file_name, map_location='cpu').to(device)
    model.eval()

    with torch.no_grad():
        root_dir = 'data/preprocess/' + ('spectrogram' if model.type == 'CNN' else 'series')
        x, y = import_data(root_dir, 'test', model.type)
        if model.type == 'RNN':
            x, y = x.transpose((1, 0, 2)), y.transpose((1, 0, 2))
        
        x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

        print('Predicting...')
        if model.type == 'RNN':
            h = model.init_hidden(ndata['test']).to(device)
            h = repackage_hidden(h)
            y_pred, _ = model(x, h)
        else:
            y_pred = model(x)

        y, y_pred = y.cpu(), y_pred.cpu()
        sz = int(y.shape[0] / ndata['test'])
        for idx in range(ndata['test']):
            if model.type == 'RNN':
                real, pred = y[:, idx], y_pred[:, idx]
            else:
                real, pred = y[idx*sz:(idx+1)*sz], y_pred[idx*sz:(idx+1)*sz]
            real, pred = real.view(-1), pred.view(-1)
            xrange = range(0, real.shape[0])
            
            if model.type == 'CNN':
                for i in range(1, real.shape[0]):
                    real[i] += real[i - 1]
                    pred[i] += pred[i - 1]

                    real[i] = max(real[i], 0)
                    pred[i] = max(pred[i], 0)
            
            loss = l1_loss(real, pred)
            print(f'Record #{idx} L1 Loss : {loss}')
            
            plt.plot(xrange, real.data, label='real value')
            plt.plot(xrange, pred.data, label='prediction')
            plt.ylabel('Weight [g]')
            plt.legend(loc=2)
            plt.savefig(f'{args.result_dir}/prediction-{idx + 1}.png', dpi=300)
            plt.cla()
        print(f'Total Test L1 Loss : {l1_loss(y, y_pred).item()}')


main()
