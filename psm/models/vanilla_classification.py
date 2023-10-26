import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchmetrics

class DenseSignalClassifier(nn.Module):
    def __init__(self, num_classes, input_dim, dense_layers, dropout_rate=0.2, batch_norm=True, 
                 activation=nn.ReLU(), l1_reg=0.01, temperature=1.0, bias=True):
        super(DenseSignalClassifier, self).__init__()

        self.temperature = temperature
        self.l1_reg = l1_reg

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
        return logits , final_layer

    def l1_regularization(self):
        return sum(p.abs().sum() for name, p in self.named_parameters() if "weight" in name) * self.l1_reg


##########

class DenseSignalClassifierModule(pl.LightningModule):
    def __init__(self,num_classes:int=20, 
                 input_dim:int=385,
                 dense_layers:list=[512, 256, 128, 64, 32], 
                 l1_reg:float=0.01,
                 dropout_rate:float=0.2, 
                 batch_norm:bool=True, 
                 activation=nn.ReLU(),
                temperature:float=1.0, 
                lr:float=0.001, 
                bias:bool=True,
                period_CosineAnnealingLR:int=20):
        super().__init__()

        self.model = DenseSignalClassifier(num_classes=num_classes,
                                             input_dim=input_dim,
                                             dense_layers=dense_layers,
                                             dropout_rate=dropout_rate,
                                             batch_norm=batch_norm,
                                             activation=activation,
                                             l1_reg=l1_reg,
                                             temperature=temperature,
                                             bias=bias)
        self.save_hyperparameters(ignore=['model','activation'])
        self.lr = lr 
        self.period_CosineAnnealingLR = period_CosineAnnealingLR
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task='multiclass',num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task='multiclass',num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task='multiclass',num_classes=num_classes)
        self.all_acc ={'train':self.train_acc,'val':self.val_acc,'test':self.test_acc}
        self.test_prediction = []
        self.test_target = []

    def forward(self,x):
        return self.model(x)
    
    def _common_step(self,batch,batch_idx,stage):
        data,target = batch
        prediction, feature = self(data)
        loss = self.criterion(prediction,target)
        # Add L1 regularization
        loss += self.model.l1_regularization()
        acc = self.all_acc[stage](prediction.argmax(dim=1),target)
        self.log(f'{stage}_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{stage}_acc', acc, on_epoch=True, prog_bar=True, logger=True)

        return {'loss':loss,'feature':feature,'target':target,'prediction':prediction}

    def training_step(self,batch,batch_idx):
        return self._common_step(batch,batch_idx,'train')
    
    def validation_step(self, batch, batch_idx):
        return self._common_step(batch,batch_idx,'val')
    
    def test_step(self, batch, batch_idx):
        output = self._common_step(batch, batch_idx, 'test')
        target = output['target'].cpu()
        prediction = output['prediction'].argmax(dim=1).cpu()
        self.test_prediction.append(prediction)
        self.test_target.append(target)
    
    def on_test_epoch_end(self):
        targets = torch.cat(self.test_target)
        predictions = torch.cat(self.test_prediction)
        self.log('test_epoch_acc', self.test_acc.compute(), prog_bar=True, logger=True,on_epoch=True)
        self.logger.experiment.log_confusion_matrix(
            y_true=targets.numpy(),
            y_predicted=predictions.numpy(),
            title="Confusion Matrix",
            row_label="Actual", column_label="Predicted",
        )
        self.test_prediction = []
        self.test_target = []


    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-07)
        scheduler = CosineAnnealingLR(optimizer, 
                                      T_max=self.period_CosineAnnealingLR, 
                                      eta_min=self.lr/self.period_CosineAnnealingLR)  
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  
            }}