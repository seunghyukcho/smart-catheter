from abc import ABC

import torch
from torch.nn import functional as F
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

from dataset import *


class BaseModel(LightningModule, ABC):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.best_score = 1e9

    def step(self, batch):
        x, y_real = batch
        y_pred = self(x)

        y_real, y_pred = y_real.view(-1), y_pred.view(-1)
        loss = F.smooth_l1_loss(y_pred, y_real)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)

        return {'loss': loss}

    def validation_epoch_end(self, outputs):
        total_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': total_loss}

        return {'val_loss': total_loss, 'log': tensorboard_logs}

    def train_dataloader(self):
        dataset = eval(f"{self.args.input_type}Dataset")(self.args.train_data_path)
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.cpu_workers
        )

    def val_dataloader(self):
        dataset = eval(f"{self.args.input_type}Dataset")(self.args.val_data_path)
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.cpu_workers
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        return optimizer
