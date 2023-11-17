#%%
import comet_ml
from pselia.config_elia import settings, load_processed_data_path, load_processed_data_path_vas
from pselia.utils import load_freq_axis
from pselia.training.callback_logger import create_callbacks_loggers, record_benchmark_results
from pselia.training.datamodule import PSDELiaDataModule , CreateTransformer, PSDNotchDataset , PSDELiaDataset_test
from pselia.eval.benchmark_vas import Benchmark_VAS
from pselia.eval.benchmark_sa import Benchmark_SA         
from pathlib import Path
import pytorch_lightning as pl
from pselia.training.dense_model import DenseSignalClassifierModule
import torch.nn as nn
from torch.utils.data import DataLoader
from pselia.training.ad_systems import AD_GMM, AD_energy

database_path = load_processed_data_path('SETTINGS1')
vas_path = load_processed_data_path_vas('SETTINGS1')
freq_axis = load_freq_axis(database_path)
freq_min , freq_max = settings.neuralnetwork.settings1.freq_range

transformer = CreateTransformer(database_path, freq_axis, freq_min=freq_min, freq_max=freq_max)
transform_psd = transformer.transform_psd
transform_label = transformer.transform_label
input_dim = transformer.dimension_psd()
ds_notch = PSDNotchDataset(database_path=vas_path, 
                        transform=transform_psd, label_transform=None)
ds_original = PSDNotchDataset(database_path=vas_path, 
                            transform=transform_psd, label_transform=None, original_psd=True)
ds_all = PSDELiaDataset_test(database_path=database_path, 
                            transform=transform_psd, label_transform=None)
 
dm = PSDELiaDataModule(database_path, batch_size=64, num_workers=1,
                        transform=transform_psd, label_transform=transform_label, val_split=0.2,
                        preload=False)
dl_notch_a = DataLoader(ds_notch, batch_size=5000, shuffle=False, num_workers=1)
dl_notch_o = DataLoader(ds_original, batch_size=500, shuffle=False, num_workers=1)
dl_all = DataLoader(ds_all, batch_size=2000, shuffle=False, num_workers=1)


dm.setup()
dl_feature= dm.ad_system_dataloader(batch_size=2000)

dense_layers = [2**k for k in range(10, 4, -1)]
hyper_params = {'input_dim':input_dim, 'dense_layers':dense_layers,
                'dropout_rate':0, 'num_direction':3,'num_face':4, 'lr':0.001,
                'bias':False, 'batch_norm':True, 'activation':nn.ReLU(), 'l1_reg':1e-4}

callbacks, logger = create_callbacks_loggers()
model = DenseSignalClassifierModule(**hyper_params)
trainer = pl.Trainer(max_epochs=2, callbacks=callbacks, logger=logger)
trainer.fit(model, dm)

ckpt= trainer.checkpoint_callback.best_model_path
best_model = DenseSignalClassifierModule.load_from_checkpoint(ckpt)
print(ckpt)
trainer.test(best_model, datamodule=dm)
ad_gmm = AD_energy(model=best_model.model)


benchmark_sa = Benchmark_SA(ad_system=ad_gmm, dl = dl_all) # Setup with proper dataloaders
result_bench_sa = benchmark_sa.evaluate_all_systems(window=1) # Evaluate and visualize
#%%
benchmarks_vas = Benchmark_VAS(ad_gmm,dl_notch_a,dl_notch_o)
result_bench_vas = benchmarks_vas.evaluate_all_sensor()


record_benchmark_results(logger, result_bench_sa,result_bench_vas)

#%%

