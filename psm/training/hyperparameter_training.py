""" Hyperparameter optimization using Optuna. 
    This script is used to optimize the hyperparameters of the DenseSignalClassifier model."""


import comet_ml
import optuna.visualization as vis

import pytorch_lightning as pl
from config import settings, create_psd_path, create_notch_path
from psm.utils.data.metadata import get_metadata_processed
from pathlib import Path
import optuna
import pandas as pd 
from psm.models.prepare_data import CreateTransformer,\
                                    PSDDataModule,PSDDataset,\
                                    PSDNotchDataset,PSDNotchDatasetOriginal,\
                                    PSDDataset_test

from psm.models.vanilla_classification import DenseSignalClassifier,\
                                                DenseSignalClassifierModule
from psm.models.callbacks_logger import create_callbacks_loggers,\
                                        record_benchmark_results
from psm.models.ad_systems import AD_GMM
from psm.eval.benchmark_sa import Benchmark_SA
from psm.eval.benchmark_vas import Benchmark_VAS
import pytorch_lightning as pl
import torch.nn as nn
from optuna.integration import PyTorchLightningPruningCallback
###

settings_proc ='SETTINGS1'
settings_simu = 'SETTINGS1'
root = Path(settings.data.path["processed"])
root_raw = Path(settings.data.path["raw"])

database_path = create_psd_path(root,settings_proc,settings_simu)
database_notch_path = create_notch_path(root,settings_proc,settings_simu)
metadata= get_metadata_processed(settings_proc, settings_simu)
resonance_avg_path = root_raw/ settings_simu / 'resonance_frequency.csv'
df_resonance_avg = pd.read_csv(resonance_avg_path)
freq_axis = metadata['freq']
### ==================== Create the transformer ==================== ###
transformer = CreateTransformer(database_path, freq=freq_axis, freq_min=0, freq_max=150)
transform_psd = transformer.transform_psd
transform_label = transformer.transform_label
input_dim = transformer.dimension_psd()

### ==================== Create the datamodule - training ==================== ###
dm = PSDDataModule(database_path, transform_psd, transform_label, batch_size=32)
dm.setup()

psd_test = PSDDataset_test(database_path, transform=transform_psd, transform_label=transform_label)
psd_notch = PSDNotchDataset(database_notch_path, transform=transform_psd, transform_label=transform_label)
psd_original = PSDNotchDatasetOriginal(database_notch_path,transform=transform_psd, transform_label=transform_label)

def train_model(hyperparams):
    """
    Train a model using PyTorch Lightning and return the performance metric.
    
    Parameters:
        hyperparams (dict): A dictionary containing the hyperparameters for the model.
        
    Returns:
        float: The performance metric of the trained model.
    """
    # Unpack hyperparameters
    input_dim = hyperparams['input_dim']
    dense_layers = hyperparams['dense_layers']
    dropout_rate = hyperparams['dropout_rate']
    num_classes = hyperparams['num_classes']
    lr = hyperparams['lr']
    batch_norm = hyperparams['batch_norm']
    activation = hyperparams['activation']
    l1_reg = hyperparams['l1_reg']
    temperature = hyperparams['temperature']
    bias = hyperparams['bias']
    period_CosineAnnealingLR = hyperparams['period_CosineAnnealingLR']
    
    # Initialize the data module
    dm = PSDDataModule(database_path, transform_psd, transform_label, batch_size=32)
    
    # Initialize the model
    model_hyperparams = {
        'input_dim': input_dim,
        'dense_layers': dense_layers,
        'dropout_rate': dropout_rate,
        'num_classes': num_classes,
        'lr': lr,
        'batch_norm': batch_norm,
        'activation': activation,
        'l1_reg': l1_reg,
        'temperature':temperature,
        'bias':bias,
        'period_CosineAnnealingLR':period_CosineAnnealingLR
    }
    model = DenseSignalClassifierModule(**model_hyperparams)
    
    # Initialize callbacks and logger
    callbacks, logger = create_callbacks_loggers()
    # add a tag for the comet experiment name
    logger.experiment.add_tag('hyperparameter_optimization')

    # Initialize the trainer
    trainer = pl.Trainer(max_epochs=40, callbacks=callbacks, logger=logger)
    
    # Train the model
    trainer.fit(model, dm)
    
    # Load the best model from checkpoint
    ckpt_path = trainer.checkpoint_callback.best_model_path
    best_model = DenseSignalClassifierModule.load_from_checkpoint(ckpt_path)
    
    # Test the model
    trainer.test(best_model, dataloaders=dm.test_dataloader())
    
    # Evaluate the model using benchmarks
    ad_system = AD_GMM(num_classes=num_classes, model=best_model.model)
    ad_system.fit(dm.train_dataloader())
    
    benchmark1 = Benchmark_SA(ad_system, psd_test, batch_size=10000)
    result_benchmark1 = benchmark1.evaluate_all_systems()
    
    benchmark2 = Benchmark_VAS(ad_system, psd_notch, psd_original, batch_size=50000, df_resonance_avg=df_resonance_avg)
    result_benchmark2 = benchmark2.evaluate_all_individus()
    
    # Extract the optimization metric
    optimization_metric, real_final_metric = record_benchmark_results(logger, result_benchmark1, result_benchmark2)

    logger.experiment.end()

    return optimization_metric, real_final_metric

def objective(trial):
    activation_map = {    'ReLU': nn.ReLU(),
    'LeakyReLU': nn.LeakyReLU(),
    'Tanh': nn.Tanh(),
    'Sigmoid': nn.Sigmoid()
        }
    # Define the hyperparameter search space
    valid_neurons = [32, 64, 128, 256, 512, 1024]
    depth = trial.suggest_int('depth', 1, 5)
    dense_layers = [valid_neurons[trial.suggest_int('dense_layers_{}'.format(i), 0, len(valid_neurons)-1)] \
                    for i in range(depth)]
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    lr = trial.suggest_float('lr', 1e-5, 1e-1,log=True)
    batch_norm = trial.suggest_categorical('batch_norm', [True, False])
    activation_name = trial.suggest_categorical('activation', ['ReLU', 'Tanh', 'Sigmoid', 'LeakyReLU'])
    activation = activation_map[activation_name]

    l1_reg = trial.suggest_float('l1_reg', 1e-5, 1e-3)
    temperature = trial.suggest_float('temperature', 1e-1,2)
    bias = trial.suggest_categorical('bias', [True, False])
    period_CosineAnnealingLR = trial.suggest_int('period_CosineAnnealingLR', 10, 100)
    
    hyperparams = {
        'input_dim': input_dim,
        'dense_layers': dense_layers,
        'dropout_rate': dropout_rate,
        'num_classes': 20,
        'lr': lr,
        'batch_norm': batch_norm,
        'activation': activation,
        'l1_reg': l1_reg,
        'temperature':temperature,
        'bias':bias,
        'period_CosineAnnealingLR':period_CosineAnnealingLR
    }

    # Train the model and get the performance metric
    performance_metric, real_final_metric = train_model(hyperparams)
    
    return performance_metric

# Create a study object and optimize the objective function.
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)


# Plotting the optimization history of the study
