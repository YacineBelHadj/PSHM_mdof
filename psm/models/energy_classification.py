from typing import Any
import torch
import torch.nn as nn
from psm.models.vanilla_classification import DenseSignalClassifierModule
import torch.nn.functional as F
from psm.models.vanilla_classification import DenseSignalClassifier
import torchmetrics

class EnergyDenseSignalClassifier(DenseSignalClassifier):
    def __init__(self, en_coef:float = 0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.en_coef = en_coef

    def get_energy(self,data):
        _, feature = self(data)
        energy = - (feature.mean(-1) - torch.logsumexp(feature, dim=-1)).mean()
        return energy


class EnergyDenseSignalClassifierModule(DenseSignalClassifierModule):
    def __init__(self, num_classes:int, input_dim:int, dense_layers:list,
                 dropout_rate:float=0.2, batch_norm:bool=True, activation=nn.ReLU(),
                 l1_reg:float=0.01, temperature:float=1.0, lr:float=0.001,
                 bias:bool=True, en_coef:float = 0.01):
        super(EnergyDenseSignalClassifierModule, self).__init__(num_classes=num_classes,
                                             input_dim=input_dim,
                                             dense_layers=dense_layers,
                                             dropout_rate=dropout_rate,
                                             batch_norm=batch_norm,
                                             activation=activation,
                                             l1_reg=l1_reg,
                                             temperature=temperature,
                                             bias=bias)
                                             
        
        # Modify the model initialization to use EnergyDenseSignalClassifier
        self.model = EnergyDenseSignalClassifier(num_classes=num_classes,
                                             input_dim=input_dim,
                                             dense_layers=dense_layers,
                                             dropout_rate=dropout_rate,
                                             batch_norm=batch_norm,
                                             activation=activation,
                                             l1_reg=l1_reg,
                                             temperature=temperature,
                                             bias=bias,
                                             en_coef=en_coef)
        
        # Then call the superclass initialization method

        self.en_coef = en_coef

        self.save_hyperparameters(ignore=['model','activation'])
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.test_prediction = []
        self.test_target = []

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
    


