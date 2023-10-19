import torch
from torch import nn
import torch.nn.functional as F

class ConvSignalClassifier(nn.Module):
    def __init__(self, num_classes: int,
                  input_dim: int,
                  conv_layers: list,
                  dense_layers: list,
                  dropout_rate: float = 0.2,
                  batch_norm: bool = True,
                  activation = nn.ReLU(),
                  l1_reg: float = 0.01,
                  temperature: float = 1.0,
                  bias: bool = True):
        
        super(ConvSignalClassifier, self).__init__()
        
        self.temperature = temperature
        self.l1_reg = l1_reg
        
        # Convolutional layers
        conv_blocks = []
        in_channels = input_dim  # Initializing with input_dim as specified
        for out_channels, kernel_size, stride in conv_layers:
            conv_blocks.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride))
            if batch_norm:
                conv_blocks.append(nn.BatchNorm1d(out_channels))
            conv_blocks.append(activation)
            conv_blocks.append(nn.Dropout(dropout_rate))
            conv_blocks.append(nn.AvgPool1d(kernel_size=2, stride=2))  # Adding Average Pooling
            in_channels = out_channels
            
        self.encoder_conv = nn.Sequential(*conv_blocks)
        
        # Global Average Pooling layer
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Dense layers
        dense_blocks = []
        in_features = in_channels  # the number of output channels from the last conv layer will be the input features for dense layers
        for units in dense_layers:
            dense_blocks.append(nn.Linear(in_features, units, bias=bias))
            if batch_norm:
                dense_blocks.append(nn.BatchNorm1d(units))
            dense_blocks.append(activation)
            dense_blocks.append(nn.Dropout(dropout_rate))
            in_features = units
        
        self.encoder_dense = nn.Sequential(*dense_blocks)
        self.classifier = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        x = self.encoder_conv(x)
        x = self.global_avg_pool(x).squeeze(-1)  # Using Global Average Pooling
        features = self.encoder_dense(x)
        logits = self.classifier(features)
        final_layer = F.softmax(logits / self.temperature, dim=1)
        
        return final_layer, logits, features
    
    def l1_regularization(self):
        return sum(p.abs().sum() for p in self.parameters()) * self.l1_reg

#

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchmetrics

class ConvClassifierModule(pl.LightningModule):
    def __init__(self, num_classes:int, input_dim:int, conv_layers:list, dense_layers:list, 
                 dropout_rate:float=0.2, batch_norm:bool=True, activation=nn.ReLU(),
                 l1_reg:float=0.01, temperature:float=1.0, lr:float=0.001,
                 bias:bool=True):
        super().__init__()

        self.model = ConvSignalClassifier(num_classes=num_classes, 
                                          input_dim=input_dim, 
                                          conv_layers=conv_layers, 
                                          dense_layers=dense_layers,
                                          dropout_rate=dropout_rate, 
                                          batch_norm=batch_norm, 
                                          activation=activation, 
                                          l1_reg=l1_reg, 
                                          temperature=temperature,
                                          bias=bias)
        self.save_hyperparameters(ignore=['model','activation'])
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
        
        # Add L1 regularization
        loss +=  self.model.l1_regularization()

        # Compute accuracy
        acc = getattr(self, f"{stage}_acc")(prediction.argmax(dim=1), target)
        self.log(f'{stage}_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log(f'{stage}_acc', acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return {'loss': loss, 'feature': feature, 'target': target, 'prediction': prediction}
    
    def on_training_epoch_end(self):
        self.log('train_epoch_acc', self.train_acc.compute(), prog_bar=True, logger=True)

    def on_validation_epoch_end(self):
        self.log('val_epoch_acc', self.val_acc.compute(), prog_bar=True, logger=True)

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        results = self._common_step(batch, batch_idx, 'val')
        val_loss = results['loss']
        self.log('val_step_loss', val_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return {'val_loss': val_loss}

    def test_step(self, batch, batch_idx):
        output = self._common_step(batch, batch_idx, 'test')
        target = output['target']
        prediction = output['prediction']
        self.test_prediction.append(prediction.argmax(dim=1))
        self.test_target.append(target)
        return {'target': target, 'prediction': prediction}
    
    def on_test_epoch_end(self):
        # Compute and log the confusion matrix
        targets = torch.cat(self.test_target)
        predictions = torch.cat(self.test_prediction)
        
        self.logger.experiment.log_confusion_matrix(
            y_true=targets,
            y_predicted=predictions,
            title="Confusion Matrix",
            row_label="Actual",
            column_label="Predicted",
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-07)
        scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=self.lr/20)  # T_max is the number of epochs after which the learning rate is restarted. Adjust accordingly.

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',  # Metric to monitor
            }
        }
