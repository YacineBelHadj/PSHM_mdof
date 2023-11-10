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
model_path.mkdir(parents=True, exist_ok=True)

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
