from typing import Any
import torch
import torch.nn as nn
from psm.models.vanilla_classification import DenseSignalClassifierModule
import torch.nn.functional as F


class EnergyDenseSignalClassifierModule(DenseSignalClassifierModule):
    def __init__(self, en_coef:float = 0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.en_coef = en_coef

    def _common_step(self, batch, batch_idx, stage):
        data, target = batch
        prediction, feature = self(data)
        loss = self.criterion(prediction, target)

        # Energy-based loss function

        energy_loss = - (feature.mean(-1) - torch.logsumexp(feature, dim=-1)).mean()
        loss += self.en_coef* energy_loss

        # Add L1 regularization
        loss += self.model.l1_regularization()

        # Compute accuracy
        acc = self.train_acc(prediction.argmax(dim=1), target)
        self.log(f'{stage}_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log(f'{stage}_acc', acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log(f'{stage}_energy_loss', energy_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return {'loss': loss,'en_loss' : energy_loss, 'feature': feature, 'target': target, 'prediction': prediction}
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        res = self._common_step(batch, batch_idx, 'test')
        return res['en_loss']

