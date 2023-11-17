from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CometLogger
from pselia.config_elia import settings
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


model_path = Path(settings.dataelia.path["model"]) 
log_path = Path(settings.local_comet["path"])

def create_callbacks_loggers(project_name_in_settings:str= 'project_elia',
                             offline:bool = False):
    """ Create callbacks and loggers for the model
    callbacks: EarlyStopping, ModelCheckpoint, LearningRateMonitor
    loggers: CometLogger
    return: List[callbacks], loggers"""
    early_stop_callback = EarlyStopping(
        monitor='val_total_loss',
        patience=50,
        verbose=False,
        mode='min'
    )
    # model path created if not exist
    model_path.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_total_loss',
        filename='best-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
        dirpath=model_path,
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    if offline:
        offline_dir = log_path
        offline_dir.mkdir(parents=True, exist_ok=True)
        logger = CometLogger(
            api_key= settings.comet['api_key'],
            workspace=settings.comet['workspace'],
            project_name=settings.comet[project_name_in_settings],
            offline=True,
            save_dir=offline_dir)
    else :
        logger = CometLogger(
            api_key=settings.comet['api_key'],
            workspace=settings.comet['workspace'],
            project_name=settings.comet[project_name_in_settings])

    res = ([early_stop_callback, checkpoint_callback, lr_monitor],
                logger)

    return res

import numpy as np
import pandas as pd
import scipy.stats as st
import os


def record_benchmark_results(logger, result_benchmark_sa, result_benchmark_vas):
    """Logs the performance metrics from the benchmarking
    AUC for VAS and SA + boxplot + contour plot
    
    Args:
    logger: the comet_ml Experiment instance
    result_benchmark1: A tuple containing (AUC scores dictionary, axs) 
    result_benchmark2: A tuple containing (Optimization metrics dictionary, axs)
    """
    print('Recording benchmark results ')
    auc_sa, boxplot_axs = result_benchmark_sa
    auc_vas, contour_axs = result_benchmark_vas
    experiment = logger.experiment

    # putting the AUC_SA in dataframe multi-indexed
    df_sa = pd.DataFrame.from_dict(auc_sa, orient='index').reset_index()
    df_sa.rename(columns={'level_0': 'face', 'level_1': 'direction'}, inplace=True)
    hm_sa_auc = st.hmean(df_sa['removal_EW916'].values)
    
    df_vas = pd.DataFrame.from_dict(auc_vas, orient='index').reset_index()
    df_vas.rename(columns={'level_0': 'face', 'level_1': 'direction'}, inplace=True)
    hm_vas_auc = st.hmean(df_vas['harmonic_mean'].values)
    
    # putting the AUC_VAS in dataframe 
    # Log the AUC VAS/SA dataframe as a table 
    auc_sa_csv = 'auc_sa_table.csv'
    df_sa.to_csv(auc_sa_csv)
    experiment.log_asset(auc_sa_csv)
    
    # Log the AUC VAS dataframe as a CSV file
    auc_vas_csv = 'auc_vas_table.csv'
    df_vas.to_csv(auc_vas_csv)
    experiment.log_asset(auc_vas_csv)
    
    # Remove the temporary CSV files
    os.remove(auc_sa_csv)
    os.remove(auc_vas_csv)
    # Calculate the mean for AUC VAS
    # Log the AUC VAS dataframe as a table
    
    # Log boxplot and contour plots
    for name, ax in boxplot_axs.items():
        experiment.log_figure(figure_name=f'boxplot_{name}', figure=ax.figure)
    
    for name, ax in contour_axs.items():
        experiment.log_figure(figure_name=f'contour_{name}', figure=ax.figure)
    

    # Log the computed metrics
    experiment.log_metrics({
        'harmonic_mean_vas': hm_vas_auc,
        'harmonic_mean_sa': hm_sa_auc,
    })

    return hm_vas_auc, hm_sa_auc
