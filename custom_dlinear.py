"""
Custom Multi-Target DLinear Implementation
===========================================

This is a simple implementation of DLinear that:
1. Works with pytorch-forecasting's TimeSeriesDataSet
2. Supports multi-target forecasting
3. Is compatible with QuantileLoss
"""

import torch
import torch.nn as nn
import lightning.pytorch as pl
from pytorch_forecasting import QuantileLoss

# ============================================================
# CUSTOM MULTI-TARGET DLINEAR IMPLEMENTATION
# ============================================================

class SeriesDecomposition(nn.Module):
    """Moving average decomposition."""
    def __init__(self, kernel_size=25):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

    def forward(self, x):
        front    = x[:, 0:1, :].repeat(1, self.padding, 1)
        end      = x[:, -1:, :].repeat(1, self.padding, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        trend    = torch.nn.functional.avg_pool1d(
            x_padded.permute(0, 2, 1),
            kernel_size=self.kernel_size, stride=1, padding=0
        ).permute(0, 2, 1)
        return x - trend, trend

class MultiTargetDLinear(pl.LightningModule):
    """Multi-target DLinear compatible with TimeSeriesDataSet."""
    def __init__(self, context_length, prediction_length, n_targets=2,
                 moving_avg=25, individual=False, loss=None, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters(ignore=['loss'])
        self.context_length    = context_length
        self.prediction_length = prediction_length
        self.n_targets         = n_targets
        self.loss_fn           = loss
        self.learning_rate     = learning_rate

        self.is_quantile_loss = isinstance(loss, QuantileLoss)
        if self.is_quantile_loss:
            self.n_quantiles = len(loss.quantiles)
            output_dim = prediction_length * self.n_quantiles
        else:
            self.n_quantiles = None
            output_dim = prediction_length

        self.decomposition = SeriesDecomposition(moving_avg)
        self.individual    = individual

        if individual:
            self.linear_seasonal = nn.ModuleList(
                [nn.Linear(context_length, output_dim) for _ in range(n_targets)])
            self.linear_trend = nn.ModuleList(
                [nn.Linear(context_length, output_dim) for _ in range(n_targets)])
        else:
            self.linear_seasonal = nn.Linear(context_length, output_dim)
            self.linear_trend    = nn.Linear(context_length, output_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1.0 / self.context_length)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        encoder_data = (torch.stack(x['encoder_target'], dim=-1)
                        if isinstance(x['encoder_target'], list)
                        else x['encoder_target'])
        batch_size = encoder_data.shape[0]
        seasonal, trend = self.decomposition(encoder_data)

        predictions = []
        for i in range(self.n_targets):
            if self.individual:
                pred = self.linear_seasonal[i](seasonal[:, :, i]) + self.linear_trend[i](trend[:, :, i])
            else:
                pred = self.linear_seasonal(seasonal[:, :, i]) + self.linear_trend(trend[:, :, i])
            if self.is_quantile_loss:
                pred = pred.view(batch_size, self.prediction_length, self.n_quantiles)
            else:
                pred = pred.view(batch_size, self.prediction_length)
            predictions.append(pred)

        return {'prediction': predictions}

    def _loss_step(self, batch):
        x, y = batch
        predictions = self(x)['prediction']
        targets = y[0] if isinstance(y[0], list) else [y[0][:, :, i] for i in range(self.n_targets)]
        return sum(self.loss_fn(p, t) for p, t in zip(predictions, targets))

    def training_step(self, batch, batch_idx):
        loss = self._loss_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._loss_step(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}}


# ============================================================
# USAGE EXAMPLE
# ============================================================

if __name__ == "__main__":
    """
    Example usage with pytorch-forecasting's TimeSeriesDataSet
    """
    from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer, EncoderNormalizer
    import pandas as pd

    # training = TimeSeriesDataSet(...)

    # Create model
    dlinear = MultiTargetDLinear(
        context_length=training.max_encoder_length,  # e.g., 24
        prediction_length=training.max_prediction_length,  # e.g., 6
        n_targets=len(training.target_names),  # e.g., 2 (temp, prcp)
        moving_avg=25,
        individual=False,
        loss=QuantileLoss(),
        learning_rate=0.001,
    )

    # Create dataloaders
    train_dataloader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)

    # Train with PyTorch Lightning
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="auto",
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=10, mode="min"),
            ModelCheckpoint(monitor="val_loss", mode="min")
        ]
    )

    trainer.fit(dlinear, train_dataloader, val_dataloader)

    # Use for predictions
    with torch.no_grad():
        for batch in val_dataloader:
            x, y = batch
            output = dlinear(x)
            predictions = output['prediction']  # List of [batch, time, quantiles]
            break