import comet_ml
from pselia.config_elia import settings, load_processed_data_path
from pselia.utils import load_freq_axis
from pselia.training.callback_logger import create_callbacks_loggers
from pselia.training.datamodule import PSDELiaDataModule , CreateTransformer
from pathlib import Path
import pytorch_lightning as pl
from pselia.training.dense_model import DenseSignalClassifierModule
import torch.nn as nn

database_path = load_processed_data_path('SETTINGS1')
freq_axis = load_freq_axis(database_path)
freq_min , freq_max = settings.neuralnetwork.settings1.freq_range

transformer = CreateTransformer(database_path, freq_axis, freq_min=freq_min, freq_max=freq_max)
transform_psd = transformer.transform_psd
transform_label = transformer.transform_label
input_dim = transformer.dimension_psd()

dm = PSDELiaDataModule(database_path, batch_size=64, num_workers=4,
                        transform=transform_psd, label_transform=transform_label, val_split=0.2,
                        preload=False)
dm.setup()
print(len(dm.train_ds), len(dm.val_ds), len(dm.test_ds))

dense_layers = [2**k for k in range(10, 4, -1)]
hyper_params = {'input_dim':input_dim, 'dense_layers':dense_layers,
                'dropout_rate':0, 'num_direction':3,'num_face':4, 'lr':0.001,
                'bias':False, 'batch_norm':True, 'activation':nn.ReLU(), 'l1_reg':1e-4}

callbacks, logger = create_callbacks_loggers()
model = DenseSignalClassifierModule(**hyper_params)
trainer = pl.Trainer(max_epochs=100, callbacks=callbacks, logger=logger)
trainer.fit(model, dm)

ckpt= trainer.checkpoint_callback.best_model_path
best_model = DenseSignalClassifierModule.load_from_checkpoint(ckpt)
print(ckpt)
trainer.test(best_model, datamodule=dm)


