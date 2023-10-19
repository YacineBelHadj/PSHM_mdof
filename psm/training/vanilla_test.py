""" In this file, we will try to train a simple model """
from config import settings, create_psd_path, create_notch_path
from psm.utils.data.metadata import get_metadata_processed
from pathlib import Path
from psm.models.prepare_data import CreateTransformer,\
                                    PSDDataModule,PSDDataset,\
                                    PSDNotchDataset,PSDNotchDatasetOriginal,\
                                    PSDDataset_test

from psm.models.vanilla_classification import DenseSignalClassifier,\
                                                DenseSignalClassifierModule
from psm.models.callbacks_logger import create_callbacks_loggers,\
                                        log_metrics
from psm.models.ad_systems import AD_GMM
from psm.eval.benchmark_sa import Benchmark_SA
from psm.eval.benchmark_vas import Benchmark_VAS


settings_proc ='SETTINGS1'
settings_simu = 'SETTINGS1'
root = Path(settings.data.path["processed"])
database_path = create_psd_path(root,settings_proc,settings_simu)
database_notch_path = create_notch_path(root,settings_proc,settings_simu)
metadata= get_metadata_processed(settings_proc, settings_simu)
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

dense_layers = [2**k for k in range(9, 4, -1)[:1]]
callbacks, logger = create_callbacks_loggers()
