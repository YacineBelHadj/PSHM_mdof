from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CometLogger
from config import settings
from pathlib import Path

model_path = Path(settings.data.path["model"])

def create_callbacks_loggers():
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

    logger = CometLogger(
        api_key=settings.comet['api_key'],
        workspace=settings.comet['workspace'],
        project_name=settings.comet['project_name'])

    res = ([early_stop_callback, checkpoint_callback, lr_monitor],
               logger)

    return res