import torch 
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchmetrics
from typing import List

class DenseSignalClassifier(nn.Module):
    """ This class is a simple dense neural network classifier.
    It is composed of a encoder and a classifier.
    it classifies the input psd into the direction and face of the sensor from 
    which it comes.
    """
    def __init__(self,input_dim:int,
                dense_layers:List[int],
                num_direction:int=3,
                num_faces:int=4,
                dropout_rate=0.2,
                batch_norm:bool=True,activation=nn.ReLU(),l1_reg:float=0.01,temperature:float=1.0,
                bias=True):
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
        self.classifier_direction = nn.Linear(in_features, num_direction)
        self.classifier_face = nn.Linear(in_features, num_faces)

    def forward(self, x):
        features = self.encoder(x)
        logits_direction = self.classifier_direction(features) 
        logits_face = self.classifier_face(features)
        prediction_direction = F.softmax(logits_direction/self.temperature, dim=1)
        prediction_face = F.softmax(logits_face/self.temperature, dim=1)
        return features ,prediction_face, prediction_direction
    
    def l1_regularization(self):
        return sum(p.abs().sum() for name, p in self.named_parameters() if "weight" in name) * self.l1_reg
    
class DenseSignalClassifierModule(pl.LightningModule):
    def __init__(self, input_dim:int = 205,
                num_direction:int=3,
                num_face:int=4,
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
        self.model = DenseSignalClassifier(input_dim=input_dim,
                                            dense_layers=dense_layers,
                                            num_direction=num_direction,
                                            num_faces=num_face,
                                            dropout_rate=dropout_rate,
                                            batch_norm=batch_norm,
                                            activation=activation,
                                            l1_reg=l1_reg,
                                            temperature=temperature,
                                            bias=bias)
        self.save_hyperparameters(ignore=['activation','model'])
        self.lr = lr
        self.period_CosineAnnealingLR = period_CosineAnnealingLR
        self.criterion = nn.CrossEntropyLoss()

        self.all_acc ={'train':{'direction':torchmetrics.Accuracy(task='multiclass',num_classes=num_direction),
                                     'face':torchmetrics.Accuracy(task='multiclass',num_classes=num_face)},
                        'val' :{'direction':torchmetrics.Accuracy(task='multiclass',num_classes=num_direction),
                                     'face':torchmetrics.Accuracy(task='multiclass',num_classes=num_face)},
                        'test':{'direction':torchmetrics.Accuracy(task='multiclass',num_classes=num_direction),
                                     'face':torchmetrics.Accuracy(task='multiclass',num_classes=num_face)}}

        self.test_prediction = {'direction':[],'face':[]}

        self.test_target = {'direction':[],'face':[]}
                       
    def forward(self, x):
        return self.model(x)
    
    def _common_step(self, batch, batch_idx, stage):
        data, true_face, true_dir = batch
        features, pred_face, pred_dir = self(data)
        loss_direction = self.criterion(pred_dir, true_dir)
        loss_face = self.criterion(pred_face, true_face)
        total_loss = loss_direction + loss_face + self.model.l1_regularization()  # Corrected line
        acc_direction = self.all_acc[stage]['direction'](pred_dir, true_dir)
        acc_face = self.all_acc[stage]['face'](pred_face, true_face)  # Corrected line
        
        show_progress = stage =='val'
        self.log(f'{stage}_loss_direction', loss_direction, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f'{stage}_loss_face', loss_face, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f'{stage}_total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f'{stage}_acc_direction', acc_direction, on_step=False, on_epoch=True, prog_bar=show_progress)
        self.log(f'{stage}_acc_face', acc_face, on_step=False, on_epoch=True, prog_bar=show_progress)

        return {'loss': total_loss, 'feature': features, 
                'true_dir': true_dir, 'true_face': true_face, 
                'pred_dir': pred_dir, 'pred_face': pred_face}

    
    def training_step(self,batch,batch_idx):
        return self._common_step(batch,batch_idx,'train')
    
    def validation_step(self, batch, batch_idx):
        return self._common_step(batch,batch_idx,'val')
    
    def test_step(self, batch, batch_idx):
        output = self._common_step(batch, batch_idx, 'test')
        target_dir = output['true_dir'].cpu()
        target_face = output['true_face'].cpu()
        prediction_dir = output['pred_dir'].argmax(dim=1).cpu()
        prediction_face = output['pred_face'].argmax(dim=1).cpu()
        self.test_prediction['direction'].append(prediction_dir)
        self.test_prediction['face'].append(prediction_face)
        self.test_target['direction'].append(target_dir)
        self.test_target['face'].append(target_face)

    def on_test_epoch_end(self) -> None:
        targets_dir = torch.cat(self.test_target['direction'])
        targets_face = torch.cat(self.test_target['face'])
        predictions_dir = torch.cat(self.test_prediction['direction'])
        predictions_face = torch.cat(self.test_prediction['face'])
        self.log('test_acc_direction', self.all_acc['test']['direction'].compute(), on_epoch=True, prog_bar=True)
        self.log('test_acc_face',self.all_acc['test']['face'].compute(), on_epoch=True, prog_bar=True)
        self.logger.experiment.log_confusion_matrix(
            y_true=targets_dir.numpy(),
            y_predicted=predictions_dir.numpy(),
            title="Confusion Matrix direction",
            row_label="Actual", column_label="Predicted")
        self.logger.experiment.log_confusion_matrix(
            y_true=targets_face.numpy(),
            y_predicted=predictions_face.numpy(),
            title="Confusion Matrix face",
            row_label="Actual", column_label="Predicted")
        self.test_prediction = {'direction':[],'face':[]}
        self.test_target = {'direction':[],'face':[]}

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
            },
        }
