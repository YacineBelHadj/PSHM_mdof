from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
# import ReduceLROnPlateau
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import LightningModule

class EnergyDenseSignalClassifier(nn.Module):
    def __init__(self, num_classes:int, input_dim:int, dense_layers:list, 
                 dropout_rate:float=0.2, batch_norm:bool=True, 
                 activation=nn.ReLU(), l1_reg:float=0.01, 
                 temperature:float=1.0, bias:bool=True, 
                 en_coef:float=0.01):
        super(EnergyDenseSignalClassifier, self).__init__()

        self.temperature = temperature
        self.l1_reg = l1_reg
        self.en_coef = en_coef

        layers = []
        in_features = input_dim
        for units in dense_layers:
            layers.append(nn.Linear(in_features, units, bias=bias))
            if batch_norm:
                layers.append(nn.BatchNorm1d(units))
            layers.append(activation)
            layers.append(nn.Dropout(dropout_rate))
            in_features = units

        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        final_layer = F.softmax(logits/self.temperature, dim=1)
        return final_layer, logits

    def l1_regularization(self):
        return sum(p.abs().sum() for p in self.parameters()) * self.l1_reg

    def get_energy(self, data,training=False):
        # only during inference
        _, logits = self(data)
        energy =  - torch.logsumexp(logits, dim=-1)
        return energy

class EnergyDenseSignalClassifierModule(LightningModule):
    def __init__(self, num_classes:int, input_dim:int, dense_layers:list, 
                 dropout_rate:float=0.2, batch_norm:bool=True, activation=nn.ReLU(),
                 l1_reg:float=0.01, temperature:float=1.0, lr:float=0.001,
                 bias:bool=True, en_coef:float = 0.01):
        super(EnergyDenseSignalClassifierModule, self).__init__()

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
        self.save_hyperparameters(ignore=['model', 'activation'])  # Matched with DenseSignalClassifierModule

        self.en_coef = en_coef
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.test_prediction = []
        self.test_target = []

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, batch_idx, stage):
        data, target = batch
        prediction, feature = self(data)
        loss = self.criterion(prediction, target)

        # Energy-based loss function
        energy_loss = self.model.get_energy(data).mean()

        loss += self.en_coef * energy_loss

        # Add L1 regularization
        loss += self.model.l1_regularization()

        # Compute accuracy
        acc = self.train_acc(prediction.argmax(dim=1), target)
        self.log(f'{stage}_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log(f'{stage}_acc', acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log(f'{stage}_energy_loss', energy_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return {'loss': loss,'en_loss' : energy_loss, 'feature': feature, 'target': target, 'prediction': prediction}

    # Define all the missing methods
    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'train')['loss']

    def validation_step(self, batch, batch_idx):
        res = self._common_step(batch, batch_idx, 'val')
        return res['loss']
    def on_trining_epoch_end(self):
        print(f'training acc: {self.train_acc.compute()}')
    def on_validation_epoch_end(self):
        print(f'validation acc: {self.val_acc.compute()}')

    def test_step(self, batch, batch_idx):
        output = self._common_step(batch, batch_idx, 'test')
        target = output['target']
        prediction = output['prediction']
        self.test_prediction.append(prediction.argmax(dim=1))
        self.test_target.append(target)
        return {'target':target, 'prediction':prediction}

    def on_test_epoch_end(self):
        targets = torch.cat(self.test_target)
        predictions = torch.cat(self.test_prediction)
        print(targets.shape, predictions.shape)
        # Assuming the logger supports log_confusion_matrix
        self.logger.experiment.log_confusion_matrix(
            y_true=targets,
            y_predicted=predictions,
            title="Confusion Matrix",
            row_label="Actual", column_label="Predicted",
        )
    
    def predict_step(self,batch, batch_idx):
        data, target = batch
        prediction, feature = self(data)
        energy = self.model.get_energy(data)
        return {'target':target, 'prediction':prediction, 'energy':energy}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-07)
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, 'min'),
            'monitor': 'val_loss',  
            'interval': 'epoch',
            'frequency': 1,
        }
        return [optimizer], [scheduler]
