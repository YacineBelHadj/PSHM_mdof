""" In this file, we will try to train a simple model """
#%% 
from config import settings, create_psd_path, create_notch_path
from psm.utils.data.metadata import get_metadata_processed
from pathlib import Path
import comet_ml
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
import pandas as pd

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
### ==================== Create the datamodule - training ==================== ###
dm = PSDDataModule(database_path, transform_psd, transform_label, batch_size=32)
dm.setup()
### ==================== Create the datasets ==================== ###





psd_test = PSDDataset_test(database_path, transform=transform_psd, transform_label=transform_label)
psd_notch = PSDNotchDataset(database_notch_path, transform=transform_psd, transform_label=transform_label)
psd_original = PSDNotchDatasetOriginal(database_notch_path,transform=transform_psd, transform_label=transform_label)

dense_layers = [2**k for k in range(9, 4, -1)]
callbacks, logger = create_callbacks_loggers()
input_dim = transformer.dimension_psd()
hyper_params = {'input_dim':input_dim, 'dense_layers':dense_layers,
                'dropout_rate':0, 'num_classes':20, 'lr':0.001,
                'batch_norm':True, 'activation':nn.ReLU(), 'l1_reg':1e-4}
model = DenseSignalClassifierModule(**hyper_params)
#%% 
trainer = pl.Trainer(max_epochs=1, callbacks=callbacks, logger=logger)
trainer.fit(model, dm)

ckpt_path = trainer.checkpoint_callback.best_model_path
best_model = DenseSignalClassifierModule.load_from_checkpoint(ckpt_path)
trainer.test(best_model, dataloaders=dm)
ad_system = AD_GMM(num_classes=20,model=best_model.model)
ad_system.fit(dm.train_dataloader())
benchmark1 = Benchmark_SA(ad_system,psd_test,batch_size=10000)
result_benchmark1 = benchmark1.evaluate_all_systems()
benchmark2 = Benchmark_VAS(ad_system,psd_notch,psd_original,batch_size=50000,df_resonance_avg=df_resonance_avg)
result_benchmark2 = benchmark2.evaluate_all_individus()
#%%
optimization_metric, real_final_metric = record_benchmark_results(logger, result_benchmark1, result_benchmark2)
