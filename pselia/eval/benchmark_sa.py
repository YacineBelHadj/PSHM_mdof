from pselia.training.datamodule import CreateTransformer, PSDELiaDataset_test
from pselia.training.dense_model import DenseSignalClassifierModule
from pselia.config_elia import settings, load_processed_data_path
from pselia.utils import load_freq_axis
from pselia.training.ad_systems import AD_GMM
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

database_path = load_processed_data_path('SETTINGS1')
freq_axis = load_freq_axis(database_path)
freq_min , freq_max = settings.neuralnetwork.settings1.freq_range

transformer = CreateTransformer(database_path, freq_axis, freq_min=freq_min, freq_max=freq_max)
transform_psd = transformer.transform_psd
transform_label = transformer.transform_label
input_dim = transformer.dimension_psd()
### data loader
ds = PSDELiaDataset_test(database_path, transform=transform_psd, label_transform=transform_label)
dl = DataLoader(ds, batch_size=2000, shuffle=False, num_workers=4)

## loader model 
model_paths = '/home/yacine/Documents/PhD/Code/GitProject/PBSHM_mdof/model/model_elia/best-epoch=99-val_loss=0.00.ckpt'
model = DenseSignalClassifierModule.load_from_checkpoint(model_paths)
ad_system = AD_GMM(num_classes=12, model=model.model)
### compute anomaly index
res ={}
for data in dl:
    res = ad_system.fit(data)
    break







