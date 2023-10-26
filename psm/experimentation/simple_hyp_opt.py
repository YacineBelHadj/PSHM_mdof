""" In this file, we will try to optimize the hyperparameters of the model
first we need to create out dataloader and datamodules
then we need to create the model
the model will be trained with the datamodule
the we use the benchmarking tool to optimize the hyperparameters
"""
import comet_ml 
import pytorch_lightning as pl
from psm.models.ad_systems import AD_GMM
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
from psm.eval.benchmark_old import Benchmark_SA,Benchmark_VAS
from psm.models.callbacks_logger import create_callbacks_loggers,\
                                        log_metrics
from psm.models.vanilla_classification import DenseSignalClassifier,\
                                                DenseSignalClassifierModule
import argparse
from functools import partial

# Settings of the simulation and the processing to use

# set the settings using parser
arg = argparse.ArgumentParser()
arg.add_argument('--settings_proc',type=str,default='SETTINGS1')
arg.add_argument('--settings_simu',type=str,default='SETTINGS1')
args = arg.parse_args()
settings_proc = args.settings_proc
settings_simu = args.settings_simu


def prepare_data():
    # Settings of the simulation and the processing to use

    root = Path(settings.data.path["processed"])
    psd_path = (root / settings_proc / settings_simu.lower()).with_suffix('.db')
    psd_vas_path = (root / settings_proc / (settings_simu.lower() + '_vas')).with_suffix('.db')
    metadata = get_metadata_processed(settings_proc, settings_simu)
    freq_axis = metadata['freq']

    create_transformer = CreateTransformer(database_path=psd_path, freq=freq_axis, freq_min=0, freq_max=150)
    transform_psd = create_transformer.transform_psd
    transform_label = create_transformer.transform_label
    input_dim = create_transformer.dimension_psd()

    dm = PSDDataModule(database_path=psd_path, transform=transform_psd, transform_label=transform_label, batch_size=64, num_workers=4)
    anomaly_ds = PSDDataset(database_path=psd_path, transform=transform_psd, transform_label=transform_label, stage='anomaly')
    test_ds = PSDDataset(database_path=psd_path, transform=transform_psd, transform_label=transform_label, stage='test')
    psd_notch = PSDNotchDataset(database_path=psd_vas_path, transform=transform_psd, transform_label=transform_label)
    psd_original = PSDNotchDatasetOriginal(database_path=psd_vas_path, transform=transform_psd, transform_label=transform_label)
    return dm, anomaly_ds, test_ds, psd_notch, psd_original, input_dim

def search_space_optuna(trial, input_dim):
    lr = trial.suggest_float('lr',1e-5,1e-1,log=True)
    droupout_rate = trial.suggest_float('droupout_rate',0.1,0.5)
    l1_reg = trial.suggest_float('l1_regul',1e-5,1e-1,log=True)
    num_layers = trial.suggest_int('num_layers',2,6)
    batch_norm = trial.suggest_categorical('batch_norm',[True,False])
    bias = trial.suggest_categorical('bias',[True,False])
    temperature = trial.suggest_float('temperature',0.1,5)
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

def train_and_evaluate(trial, dm, anomaly_ds, test_ds, psd_notch, psd_original, input_dim):
    hyper_params = search_space_optuna(trial, input_dim)
    dm, anomaly_ds, test_ds, psd_notch, psd_original, input_dim = prepare_data()

    model = DenseSignalClassifierModule(**hyper_params)
    callbacks, logger = create_callbacks_loggers(project_name_in_settings='project_name1', offline=True)
    logger.experiment.set_name('simple+T')
    trainer = pl.Trainer(max_epochs=60, logger=logger, callbacks=callbacks)
    trainer.fit(model, dm)
    ckpt_path = trainer.checkpoint_callback.best_model_path
    best_model = DenseSignalClassifierModule.load_from_checkpoint(ckpt_path)
    logger.experiment.log_model('best_model', ckpt_path)
    logger.experiment.log_asset(ckpt_path)
    ad_system = AD_GMM(num_classes=20,model=best_model.model)
    ad_system.fit(dm.train_dataloader())
    benchmark1 = Benchmark_SA(ad_system,anomaly_ds,test_ds,batch_size=2**14)
    test1 = benchmark1.evaluate()
    benchmark2 = Benchmark_VAS(ad_system,psd_notch,psd_original,batch_size=2**14)
    test2 = benchmark2.evaluate_all_systems()
    opt_metric,auc = log_metrics(logger,test1,test2)


    # close the logger
    logger.experiment.end()
    return opt_metric

def main():
    dm, anomaly_ds, test_ds, psd_notch, psd_original, input_dim = prepare_data()
    optimazation = partial(train_and_evaluate,
                            dm=dm,
                            anomaly_ds=anomaly_ds, 
                           test_ds=test_ds, 
                           psd_notch=psd_notch,
                            psd_original=psd_original,
                            input_dim=input_dim)
    # fix the seed for reproducibility
    TPE_sampler = optuna.samplers.TPESampler(seed=10)

    study = optuna.create_study(direction='maximize',sampler=TPE_sampler)
    study.optimize(optimazation, n_trials=100,gc_after_trial=True)

if __name__ == '__main__':
    main()
    
