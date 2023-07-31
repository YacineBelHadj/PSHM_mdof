import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics

class DenseSignalClassifier(nn.Module):
    def __init__(self, num_classes:int,
                  input_dim:int, 
                  dense_layers:list, 
                  dropout_rate:float=0.2, 
                 batch_norm:bool=True, 
                 activation=nn.ReLU(), l1_reg:float=0.01, 
                 temperature:float=1.0):
        super(DenseSignalClassifier, self).__init__()

        self.temperature = temperature
        self.l1_reg = l1_reg

        layers = []
        in_features = input_dim
        for units in dense_layers:
            layers.append(nn.Linear(in_features, units))
            if batch_norm:
                layers.append(nn.BatchNorm1d(units))
            layers.append(activation)
            layers.append(nn.Dropout(dropout_rate))
            in_features = units

        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x ):
        features = self.encoder(x)
        logits = self.classifier(features) 
        final_layer = F.softmax(logits, dim=1)
        return final_layer, logits
       
        

    def l1_regularization(self):
        return sum(p.abs().sum() for p in self.parameters())*self.l1_reg

##########


class DenseSignalClassifierModule(pl.LightningModule):
    def __init__(self, num_classes:int, input_dim:int, dense_layers:list, 
                 dropout_rate:float=0.2, batch_norm:bool=True, activation=nn.ReLU(),
                 l1_reg:float=0.01, temperature:float=1.0, lr:float=0.001):
        super().__init__()

        self.model = DenseSignalClassifier(num_classes=num_classes, 
                                           input_dim=input_dim, 
                                           dense_layers=dense_layers, 
                                           dropout_rate=dropout_rate, 
                                           batch_norm=batch_norm, 
                                           activation=activation,
                                           l1_reg=l1_reg, 
                                           temperature=temperature)
        self.save_hyperparameters(ignore=['model'])
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
        prediction,feature = self(data)
        loss = self.criterion(prediction, target)
        
        # Add L1 regularization
        loss +=  self.model.l1_regularization()

        # Compute accuracy
        acc = self.train_acc(prediction.argmax(dim=1), target)
        self.log(f'{stage}_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{stage}_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'feature':feature, 'target':target, 'prediction':prediction}



    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'val')['loss']

    def test_step(self, batch, batch_idx):
        output = self._common_step(batch, batch_idx, 'test')
        target = output['target']
        prediction = output['prediction']
        self.test_prediction.append(prediction.argmax(dim=1))
        self.test_target.append(target)
        return {'target':target, 'prediction':prediction}
    
    def on_test_epoch_end(self):
        # Compute and log the confusion matrix
        targets = torch.cat(self.test_target)
        predictions = torch.cat(self.test_prediction)
        print(targets.shape, predictions.shape)
        
        self.logger.experiment.log_confusion_matrix(
            y_true=targets,
            y_predicted=predictions,
            title="Confusion Matrix",
            row_label="Actual", column_label="Predicted",
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-07)
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, 'min'),
            'monitor': 'val_loss',  # Metric to monitor
            'interval': 'epoch',  # The scheduler will be updated after each epoch
            'frequency': 1,  # The frequency of the scheduler updates in number of epochs
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    
