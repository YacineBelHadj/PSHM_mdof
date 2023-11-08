from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CometLogger
from config import settings
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


model_path = Path(settings.data.path["model"])
log_path = Path(settings.local_comet["path"])

def create_callbacks_loggers(project_name_in_settings:str= "project_name2",
                             offline:bool = False):
    """ Create callbacks and loggers for the model
    callbacks: EarlyStopping, ModelCheckpoint, LearningRateMonitor
    loggers: CometLogger
    return: List[callbacks], loggers"""
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=50,
        verbose=False,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='best-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
        dirpath=model_path/'model',
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


def record_benchmark_results(logger, result_benchmark1, result_benchmark2):
    """Logs the performance metrics from the benchmarking
    AUC for VAS and SA + boxplot + contour plot
    
    Args:
    logger: the comet_ml Experiment instance
    result_benchmark1: A tuple containing (AUC scores dictionary, axs) 
    result_benchmark2: A tuple containing (Optimization metrics dictionary, axs)
    """
    print('Recording benchmark results ')
    auc_sa, boxplot_axs = result_benchmark1
    auc_vas, contour_axs = result_benchmark2
    experiment = logger.experiment

    # putting the AUC_SA in dataframe multi-indexed
    auc_sa_df = pd.concat(dict(auc_sa),axis=1)
    anomaly_level_data = auc_sa_df.xs('anomaly_level', level=1, axis=1).mean(axis=1)
    auc_sa_df = auc_sa_df.set_index(anomaly_level_data)
    auc_sa_df = auc_sa_df.xs('anomaly_index', level=1, axis=1)
    auc_sa_df.index = np.round(auc_sa_df.index, 4)
    rows_003 = auc_sa_df.loc[0.03]


    # Log the AUC SA dataframe as a table
    

    
    # putting the AUC_VAS in dataframe 
    auc_vas_df = pd.DataFrame(auc_vas)
    # Log the AUC VAS/SA dataframe as a table 
    auc_sa_csv = 'auc_sa_table.csv'
    auc_sa_df.to_csv(auc_sa_csv)
    experiment.log_asset(auc_sa_csv, file_name=auc_sa_csv)
    
    # Log the AUC VAS dataframe as a CSV file
    auc_vas_csv = 'auc_vas_table.csv'
    auc_vas_df.to_csv(auc_vas_csv)
    experiment.log_asset(auc_vas_csv, file_name=auc_vas_csv)
    
    # Remove the temporary CSV files
    os.remove(auc_sa_csv)
    os.remove(auc_vas_csv)
    # Calculate the mean for AUC VAS
    means_auc_vas = auc_vas_df.mean(axis=1).to_dict()
    means_auc_vas = {f'{key}_vas': value for key, value in means_auc_vas.items()}
    
    # Log the AUC VAS dataframe as a table
    
    # Log boxplot and contour plots
    for name, ax in boxplot_axs.items():
        experiment.log_figure(figure_name=f'boxplot_{name}', figure=ax.figure)
    
    for name, ax in contour_axs.items():
        experiment.log_figure(figure_name=f'contour_{name}', figure=ax.figure)
    

    # Log the computed metrics
    experiment.log_metrics({
        "mean_auc_sa_003": np.mean(rows_003),
        "hmean_auc_sa_003": st.hmean(rows_003),
        "gmean_auc_sa_003": st.gmean(rows_003),
        **means_auc_vas
    })
    optimization_metric = means_auc_vas['harmonic_mean_vas'] 
    goal_metric = np.mean(rows_003)
    return optimization_metric, goal_metric
