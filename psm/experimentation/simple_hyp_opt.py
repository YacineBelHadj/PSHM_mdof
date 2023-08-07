""" In this file, we will try to optimize the hyperparameters of the model
first we need to create out dataloader and datamodules
then we need to create the model
the model will be trained with the datamodule
the we use the benchmarking tool to optimize the hyperparameters
"""
import comet_ml 
import pytorch_lightning as pl
from psm.models.ad_gmm import AD_GMM
import torch.nn as nn 
import torch
import optuna
import pandas as pd
from pathlib import Path
from config import settings
from psm.utils.data.metadata import get_metadata_processed
from psm.models.prepare_data import CreateTransformer,\
                                        PSDDataModule,PSDDataset,\
                                        PSDNotchDataset,PSDNotchDatasetOriginal
from psm.eval.benchmark import Benchmark_SA,Benchmark_VAS
from psm.models.callbacks_logger import create_callbacks_loggers,\
                                        log_metrics
from psm.models.vanilla_classification import DenseSignalClassifier,\
                                                DenseSignalClassifierModule

# Settings of the simulation and the processing to use
settings_proc = 'SETTINGS1'
settings_simu = 'SETTINGS1'
# load the neccessay paths + frequency axis
root = Path(settings.data.path["processed"])
psd_path = (root /settings_proc/settings_simu.lower()).with_suffix('.db')
psd_vas_path = (root /settings_proc/(settings_simu.lower()+'_vas')).with_suffix('.db')
metadata = get_metadata_processed(settings_proc, settings_simu)
freq_axis = metadata['freq']
# Now let's create our datamodules and use the correct transformer
create_transformer = CreateTransformer(database_path=psd_path,
                                       freq=freq_axis,
                                       freq_min=0,
                                       freq_max=150)
transform_psd = create_transformer.transform_psd
transform_label = create_transformer.transform_label
input_dim = create_transformer.dimension_psd()
# now that we have our preprocessing step for the PSD
# let's create our dataModule that define what is a training testing
# and validation dataloader , we also define the anomalous data loading 
dm = PSDDataModule(database_path = psd_path,
                   transform=transform_psd,
                   transform_label= transform_label,
                   batch_size= 64)
anomaly_ds = PSDDataset(database_path = psd_path,
                        transform=transform_psd,
                        transform_label = transform_label,
                        stage='anomaly')
test_ds = PSDDataset(database_path = psd_path,
                     transform=transform_psd,
                     transform_label = transform_label,
                     stage='test')
psd_notch =PSDNotchDataset(database_path=psd_vas_path,
                           transform = transform_psd,
                           transform_label= transform_label)

psd_original = PSDNotchDatasetOriginal(database_path=psd_vas_path,
                                       transform = transform_psd,
                                       transform_label= transform_label)

VAS_auc = []
SA_auc = []
# define the search space for optuna
def search_space_optuna(trial):
    lr = trial.suggest_float('lr',1e-5,1e-1,log=True)
    droupout_rate = trial.suggest_uniform('droupout_rate',0.1,0.5)
    l1_reg = trial.suggest_float('l1_regul',1e-5,1e-1,log=True)
    num_layers = trial.suggest_int('num_layers',2,6)
    batch_norm = trial.suggest_categorical('batch_norm',[True,False])
    bias = trial.suggest_categorical('bias',[True,False])
    temperature = trial.suggest_uniform('temperature',0.1,5)
    dense_layers = [2**k for k in range(9,4,-1)[:num_layers]]
    hyper_params = {'input_dim':input_dim,
                    'num_classes':20,
                    'dense_layers':dense_layers,
                    'dropout_rate':droupout_rate,
                    'batch_norm':batch_norm,
                    'bias':bias,
                    'temperature':temperature,
                    'batch_norm':batch_norm,
                    'activation':nn.ReLU(),
                    'l1_reg':l1_reg,
                    'bias':bias,
                    'lr':lr}
    return hyper_params

# define the objective function for optuna to optimize
def objective(trial):
    hyper_params = search_space_optuna(trial)
    
    model = DenseSignalClassifierModule(**hyper_params)
    callbacks,logger = create_callbacks_loggers(project_name_in_settings='project_name1')
    trainer = pl.Trainer(max_epochs=60,
                        logger=logger,
                        callbacks=callbacks)
    trainer.fit(model, dm)
    ckpt_path = trainer.checkpoint_callback.best_model_path
    best_model = DenseSignalClassifierModule.load_from_checkpoint(ckpt_path)
    logger.experiment.log_model('best_model',ckpt_path)
    logger.experiment.log_asset(ckpt_path)
    
    ad_system = AD_GMM(num_classes=20,model=best_model.model)
    ad_system.fit(dm.train_dataloader())
    benchmark1 = Benchmark_SA(ad_system,anomaly_ds,test_ds,batch_size=2**14)
    test1 = benchmark1.evaluate()
    benchmark2 = Benchmark_VAS(ad_system,psd_notch,psd_original,batch_size=2**14)
    test2 = benchmark2.evaluate_all_systems()
    opt_metric,auc = log_metrics(logger,test1,test2)
    VAS_auc.append(auc)
    SA_auc.append(opt_metric)
    logger.experiment.add_tags('opt')
    logger.experiment.add_tags('simple+T')
    return opt_metric

study = optuna.create_study(direction='maximize')
study.optimize(objective,n_trials=100)

import matplotlib.pyplot as plt
plt.plot(VAS_auc,SA_auc,'o')
plt.xlabel('VAS')
plt.ylabel('SA')
plt.show()




