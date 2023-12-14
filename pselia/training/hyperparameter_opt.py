import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import comet_ml
import pytorch_lightning as pl
from pselia.config_elia import load_processed_data_path, load_processed_data_path_vas, load_optimazation_path
from pselia.utils import load_freq_axis
from pselia.training.datamodule import CreateTransformer, PSDELiaDatasetBuilder, PSDELiaDataModule
import optuna
import pandas as pd
from pselia.training.dense_model import DenseSignalClassifierModule
from pselia.training.callback_logger import create_callbacks_loggers, record_benchmark_results
from pselia.training.ad_systems import AD_GMM
from pselia.eval.benchmark_vas import Benchmark_VAS
from pselia.eval.benchmark_sa import Benchmark_SA
import torch.nn as nn
from torch.utils.data import DataLoader

settings_proc = 'SETTINGS3'

database_path = load_processed_data_path(settings_proc)
vas_path = load_processed_data_path_vas(settings_proc)
freq_axis = load_freq_axis(database_path)

transformer = CreateTransformer(database_path, freq_axis, freq_min=0, freq_max=50)
transform_psd = transformer.transform_psd
transform_face = transformer.transform_face
transform_direction = transformer.transform_direction

input_dim = transformer.dimension_psd()

# Build the dataset for Notch data
ds_notch = PSDELiaDatasetBuilder()\
    .set_database_path(vas_path)\
    .set_transform_psd(transform_psd)\
    .set_transform_face(transform_face)\
    .set_transform_direction(transform_direction)\
    .set_table_name('VAS_NOTCH')\
    .set_columns(['psd', 'direction', 'face', 'aff_f', 'aff_amp', 'date_time']).build()

# Build the dataset for Original PSD data
ds_original = PSDELiaDatasetBuilder()\
    .set_database_path(vas_path)\
    .set_transform_psd(transform_psd)\
    .set_table_name('ORIGINAL_PSD')\
    .set_columns(['PSD', 'direction', 'face', 'aff_f', 'aff_amp', 'date_time']).build()


# Build the dataset for all data
ds_all = PSDELiaDatasetBuilder()\
    .set_database_path(database_path)\
    .set_transform_psd(transform_psd)\
    .set_columns(['PSD', 'direction', 'face', 'date_time', 'stage', 'anomaly_description']).build()




ds_feature = PSDELiaDatasetBuilder()\
    .set_database_path(database_path)\
    .set_transform_psd(transform_psd)\
    .add_condition("stage=?", ['training'])\
    .set_columns(['psd']).build()


# Create DataLoader for feature dataset

# DataModule
dm = PSDELiaDataModule(database_path, batch_size=64, num_workers=4,
                        transform_psd=transform_psd, transform_direction = transform_direction,
                        transform_face= transform_face, val_split=0.2,
                        preload=True)

# DataLoaders
dl_notch_a = DataLoader(ds_notch, batch_size=10000, shuffle=False, num_workers=1)
dl_notch_o = DataLoader(ds_original, batch_size=10000, shuffle=False, num_workers=1)
dl_all = DataLoader(ds_all, batch_size=2000, shuffle=False, num_workers=1)
dl_feature = DataLoader(ds_feature, batch_size=2000, shuffle=False, num_workers=1)
dm.setup()

def train_model(hyperparams):
    model = DenseSignalClassifierModule(**hyperparams)
    callbacks, logger = create_callbacks_loggers(project_name_in_settings='project_elia_2')
    logger.experiment.add_tag('hyperparameter_optimization')
    # log model anomaly index as a hyperparameter
    # log hyperparameters
    logger.experiment.log_parameters(hyperparams)
    trainer = pl.Trainer(max_epochs=50, callbacks=callbacks, logger=logger)
    trainer.fit(model, dm)

    ckpt= trainer.checkpoint_callback.best_model_path
    best_model = DenseSignalClassifierModule.load_from_checkpoint(ckpt)
    trainer.test(best_model, datamodule=dm)

    ad = AD_GMM(num_classes=12, model=best_model.model)
    ad.fit(dl_feature)

    
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
    activation_map = { 'ReLU': nn.ReLU(),
    'LeakyReLU': nn.LeakyReLU(),
    'Tanh': nn.Tanh(),
    'Sigmoid': nn.Sigmoid()
        }
    valid_neurons_1= [256,512,1024,2048]
    valid_neurons_2 = [32,64,128,256,512,1024]
    depth = trial.suggest_int('depth', 2, 6)
    dense_layers = []
    for i in range(depth):
        if i == 0:
            dense_layers.append(trial.suggest_categorical(f'neurons_{i}', valid_neurons_1))
        else:
            dense_layers.append(trial.suggest_categorical(f'neurons_{i}', valid_neurons_2))

    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    lr = trial.suggest_float('lr', 1e-5, 1e-1,log=True)
    batch_norm = trial.suggest_categorical('batch_norm', [True, False])
    activation_name = trial.suggest_categorical('activation', ['ReLU', 'Tanh', 'Sigmoid', 'LeakyReLU'])
    activation = activation_map[activation_name]
    l1_reg = trial.suggest_float('l1_reg', 1e-6, 1e-2, log=True)
    temperature = trial.suggest_float('temperature', 1e-1,3)
    bias = trial.suggest_categorical('bias', [True, False])
    period_CosineAnnealingLR = trial.suggest_int('period_CosineAnnealingLR', 10, 100)
    hyperparams = {'input_dim':input_dim, 'dense_layers':dense_layers,
                    'dropout_rate':dropout_rate, 'num_direction':3,'num_face':4, 'lr':lr,
                    'bias':bias, 'batch_norm':batch_norm, 'activation':activation, 'l1_reg':l1_reg,
                    'temperature':temperature, 'period_CosineAnnealingLR':period_CosineAnnealingLR}
    res = train_model(hyperparams)
    return res['vas_auc']

opt_path = load_optimazation_path()
study_name = 'hyperparameter_optimization_nperseg4'
# create path to store the study
path_db_study = opt_path / f'optuna_studyname_{study_name}.db'
path_db_study.parent.mkdir(parents=True, exist_ok=True)
# get str from path
path_db_study = str(path_db_study)
storage_name = f'sqlite:///{path_db_study}'
study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage_name)
study.optimize(objective, n_trials=100)


    
        
