import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from src.model.model import UNET


class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.unet = UNET()

    def forward(self, x):
        out = self.unet(x)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        x_hat = self.forward(x)  
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        x_hat = self.forward(x)  
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss)

