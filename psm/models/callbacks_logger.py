from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CometLogger
from config import settings
from pathlib import Path
import pandas as pd

model_path = Path(settings.data.path["model"])
log_path = Path(settings.local_comet["path"])

def create_callbacks_loggers(project_name_in_settings:str= 'project_name1',
                             offline:bool = False):
    """ Create callbacks and loggers for the model
    callbacks: EarlyStopping, ModelCheckpoint, LearningRateMonitor
    loggers: CometLogger
    return: List[callbacks], loggers"""
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=20,
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

def log_metrics(logger,test1,test2):
    df_perf = pd.DataFrame(test2['individual_metric'])
    df_auc=pd.DataFrame(test1[0.03])
    df_auc.columns = ['AUC_0.03']
    df_perf = pd.concat([df_perf,df_auc.T],axis=0)
    # log the dataframe in html and csv format
    logger.experiment.log_html(df_perf.to_html())
    logger.experiment.log_asset_data(df_perf.to_csv(),name="aucs.csv")
    # log global metrics as metrics
    df_mean = df_perf.mean(axis=1)
    for key in df_mean.keys():
        logger.experiment.log_metric('mean_'+key,df_mean[key])
    # log the axs 
    for n,ax in test2['axs'].items():
        logger.experiment.log_figure(figure_name=n,figure=ax.get_figure())
    return df_mean['weighted_auc_VAS'], df_mean['AUC_0.03']