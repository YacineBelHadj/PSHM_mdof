import optuna
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import comet_ml
import pytorch_lightning as pl
from pselia.config_elia import settings, load_processed_data_path, load_processed_data_path_vas
from pselia.utils import load_freq_axis
from pselia.training.datamodule import PSDELiaDataModule , CreateTransformer, PSDNotchDataset , PSDELiaDataset_test
import optuna
import pandas as pd 
from pselia.training.dense_model import DenseSignalClassifierModule
from pselia.training.callback_logger import create_callbacks_loggers, record_benchmark_results
from pselia.training.ad_systems import AD_GMM, AD_energy
from pselia.eval.benchmark_vas import Benchmark_VAS
from pselia.eval.benchmark_sa import Benchmark_SA
import torch.nn as nn
from torch.utils.data import DataLoader
from argparse import ArgumentParser
#let's add logging 
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler("logs/hyperparameter_optimization.log")])

parser = ArgumentParser(description='hyperparameter Optimization')
parser.add_argument('--settings',type=str,\
                    help='settings_param name in configuration/settings_elia.toml'\
                        ,default='SETTINGS3')
args = parser.parse_args()
settings_proc = args.settings
logging.info(f'hyperparameter optimization for {settings_proc}')

database_path = load_processed_data_path(settings_proc)
vas_path = load_processed_data_path_vas(settings_proc)
freq_axis = load_freq_axis(database_path)

transformer = CreateTransformer(database_path, freq_axis, freq_min=0, freq_max=50)
transform_psd = transformer.transform_psd
transform_label = transformer.transform_label
input_dim = transformer.dimension_psd()

ds_notch = PSDNotchDataset(database_path=vas_path, 
                        transform=transform_psd, label_transform=None)
ds_original = PSDNotchDataset(database_path=vas_path, 
                            transform=transform_psd, label_transform=None, original_psd=True)
ds_all = PSDELiaDataset_test(database_path=database_path, 
                            transform=transform_psd, label_transform=None)
logging.info(f'number of samples in ds_notch: {len(ds_notch)}')
dm = PSDELiaDataModule(database_path, batch_size=64, num_workers=4,
                        transform=transform_psd, label_transform=transform_label, val_split=0.2,
                        preload=False)
dl_notch_a = DataLoader(ds_notch, batch_size=10000, shuffle=False, num_workers=1)
dl_notch_o = DataLoader(ds_original, batch_size=10000, shuffle=False, num_workers=1)
dl_all = DataLoader(ds_all, batch_size=2000, shuffle=False, num_workers=1)

logging.info(f'number of samples in dl_notch_a: {len(dl_notch_a)}')
dm.setup()
dl_feature= dm.ad_system_dataloader(batch_size=2000)

def train_model(hyperparams):
    anomaly_index = hyperparams.pop('anomaly_index')
    model = DenseSignalClassifierModule(**hyperparams)
    callbacks, logger = create_callbacks_loggers(project_name_in_settings='project_elia_2')
    logger.experiment.add_tag('hyperparameter_optimization')
    # log model anomaly index as a hyperparameter
    logger.experiment.log_parameter('type_of_AI', anomaly_index)
    # log hyperparameters
    logger.experiment.log_parameters(hyperparams)
    trainer = pl.Trainer(max_epochs=50, callbacks=callbacks, logger=logger)
    trainer.fit(model, dm)

    ckpt= trainer.checkpoint_callback.best_model_path
    best_model = DenseSignalClassifierModule.load_from_checkpoint(ckpt)
    trainer.test(best_model, datamodule=dm)

    if anomaly_index == 'energy':
        ad = AD_energy(model=best_model.model)
    elif anomaly_index == 'gmm':
        ad = AD_GMM(num_classes=12, model=best_model.model)
        ad.fit(dl_feature)
    else:
        raise ValueError(f"anomaly_index {anomaly_index} not supported")
    
    benchmark_sa = Benchmark_SA(ad_system=ad, dl = dl_all) # Setup with proper dataloaders
    result_bench_sa = benchmark_sa.evaluate_all_systems(window=1) # Evaluate and visualize
    #%%
    benchmarks_vas = Benchmark_VAS(ad,dl_notch_a,dl_notch_o)
    result_bench_vas = benchmarks_vas.evaluate_all_sensor()


    vas_auc, sa_auc = record_benchmark_results(logger, result_bench_sa,result_bench_vas)
    logger.experiment.log_metric('vas_auc', vas_auc)
    logger.experiment.log_metric('sa_auc', sa_auc)
    logger.experiment.end()
    plt.close('all')
    return {'vas_auc':vas_auc, 'sa_auc':sa_auc}

def objective(trial):
    valid_neurons= [64,256,512,1024,2048,4096]
    depth = trial.suggest_int('depth', 3, 6)
    dense_layers = [valid_neurons[trial.suggest_int('dense_layers_{}'.format(i), 0, len(valid_neurons)-1)] \
                    for i in range(depth)]
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.2)
    lr = trial.suggest_float('lr', 1e-5, 1e-1,log=True)

    l1_reg = trial.suggest_float('l1_reg', 1e-6, 1e-2, log=True)
    temperature = trial.suggest_float('temperature', 1e-1,10, log=True)
    anomaly_index = trial.suggest_categorical('anomaly_index', ['energy', 'gmm'])
    period_CosineAnnealingLR = trial.suggest_int('period_CosineAnnealingLR', 10, 100)
    hyperparams = {'input_dim':input_dim, 'dense_layers':dense_layers,
                    'dropout_rate':dropout_rate, 'num_direction':3,'num_face':4, 'lr':lr,
                    'bias':True, 'batch_norm':True, 'activation':nn.LeakyReLU(), 'l1_reg':l1_reg,
                    'anomaly_index':anomaly_index, 'temperature':temperature, 'period_CosineAnnealingLR':period_CosineAnnealingLR}
    res = train_model(hyperparams)
    return res['vas_auc']

study_name = 'hyperparameter_optimization_3'

storage_name = f'sqlite:///optuna_studyname_{study_name}.db'
logging.info(f'hyperparameter optimization for {settings_proc} with study name {study_name}')
study = optuna.create_study(study_name=study_name, storage=storage_name, direction="maximize", load_if_exists=True)
logging.info(f'hyperparameter optimization for {settings_proc} with study name {study_name} and running optimize')
study.optimize(objective, n_trials=50) 