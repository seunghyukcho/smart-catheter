from abc import ABC

import torch
from torch.nn import functional as F
from pytorch_lightning import LightningModule, TrainResult, EvalResult
from torch.utils.data import DataLoader

from dataset import *


class BaseModel(LightningModule, ABC):

    def __init__(self, options):
        super().__init__()
        self.save_hyperparameters()
        self.options = options

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred, y_true = self(x).view(-1), y_true.view(-1)
        loss = F.smooth_l1_loss(y_pred, y_true)

        result = TrainResult(loss)
        result.log("train_loss", loss, prog_bar=True)
        return result

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred, y_true = self(x).view(-1), y_true.view(-1)
        loss = F.l1_loss(y_pred, y_true).data

        result = EvalResult(checkpoint_on=loss, early_stop_on=loss)
        result.log("val_metric", loss, prog_bar=True)
        return result

    def train_dataloader(self):
        dataset = eval(f"{self.options.input_type}Dataset")(self.options.train_data_path)
        return DataLoader(
            dataset,
            batch_size=self.options.batch_size,
            shuffle=True,
            num_workers=self.options.cpu_workers
        )

    def val_dataloader(self):
        dataset = eval(f"{self.options.input_type}Dataset")(self.options.val_data_path)
        return DataLoader(
            dataset,
            batch_size=self.options.batch_size,
            shuffle=False,
            num_workers=self.options.cpu_workers
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.options.lr)
        return optimizer
