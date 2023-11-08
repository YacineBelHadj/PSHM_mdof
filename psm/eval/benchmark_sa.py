""" In this file, we will implement the benchmark for our anomaly detection method. 
the method is evalutad on the strcutural anomaly dataset.
"""
import numpy as np
from config import settings
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from psm.models.ad_systems import AD_system
from dataclasses import dataclass
from typing import Union
from functools import partial
from psm.eval.aucs_computation import compute_auc
from collections import defaultdict

def select_system(df,system_name):
    return df[df['system_name']==system_name]

def order_df(df_sys):
    stage_order = ['train', 'test','anomaly']
    df_sys.loc[:,'stage'] = pd.Categorical(df_sys['stage'], categories=stage_order, ordered=True)
    df_sys = df_sys.sort_values(by=['stage', 'anomaly_level'])
    df_sys.loc[:,'stage']= df_sys['stage'].astype(str)
    df_sys = df_sys.reset_index(drop=True)
    return df_sys

def get_CL(df_sys):
    data_tr = df_sys['anomaly_index'][df_sys['stage']=='train']
    data_tr = data_tr[~data_tr.isna()]
    UCL = np.quantile(data_tr, 0.95)
    LCL = np.quantile(data_tr, 0.05)
    return UCL, LCL

def scale_anomaly_index(df_sys):
    df_sys.loc[:,'anomaly_index'] = -1 * df_sys['anomaly_index']
    # add the min to make it positive
    df_sys.loc[:,'anomaly_index'] = df_sys['anomaly_index'] - df_sys['anomaly_index'].min() + 1
    return df_sys

def get_auc_per_level(df_sys):
    test_data = df_sys['anomaly_index'][df_sys['stage'] == 'test']
    anoumalous_data = df_sys[df_sys['stage'] == 'anomaly']
    compute_auc_partial = partial(compute_auc, healthy=test_data)
    df_grouped = anoumalous_data.groupby(['anomaly_level'])['anomaly_index'].apply(compute_auc_partial)
    df_grouped = df_grouped.reset_index()
    return df_grouped

def average_window(df_sys, window=20):
    # Apply rolling median only within each 'stage' and 'anomaly_level' group
    df_sys['anomaly_index'] = df_sys.groupby(['stage', 'anomaly_level'])['anomaly_index'].transform(lambda x: x.rolling(window).median())
    return df_sys

def compute_anomaly_index(psd_dl:DataLoader,
                          ad_system:AD_system):
    lists = {
        'system_name': [],
        'anomaly_index': [],
        'anomaly_level': [],
        'stage': [],
        'excitation': [],
        'latent': []}

    for batch in psd_dl:
        psd, system_name, anomaly_level, stage, excitation_amplitude, latent_value = batch
        anomaly_index = ad_system.predict(psd)

        for key, value in zip(lists.keys(), [system_name, anomaly_index, anomaly_level, stage, excitation_amplitude, latent_value]):
            if isinstance(value, tuple):
                lists[key].extend(list(value))
            else:
                lists[key].extend(value.tolist())
    df_res = pd.DataFrame(lists)
    return df_res

def plot_boxplot_with_train(df_sys, UCL,system_name:int=None):
    # Add a dummy anomaly level for train data to plot it alongside test data
    train_data = df_sys[df_sys['stage'] == 'train'].copy()
    train_data['anomaly_level'] = 'train'
    combined_df = pd.concat([train_data, df_sys[df_sys['stage'] != 'train']])
    
    # Convert all anomaly levels to strings and rename '0.0' to 'test'
    combined_df['anomaly_level'] = combined_df['anomaly_level'].astype(str)
    combined_df['anomaly_level'] = combined_df['anomaly_level'].replace('0.0', 'test')

    # Sort categories
    categories_order = ['train', 'test'] + sorted([str(x) for x in df_sys['anomaly_level'].unique() if x != 0.0])
    combined_df['anomaly_level'] = pd.Categorical(combined_df['anomaly_level'], categories=categories_order, ordered=True)

    # Boxplot
    fig, ax = plt.subplots(figsize=(8, 6))
    combined_df.boxplot(column='anomaly_index', by='anomaly_level', ax=ax, showfliers=False)
    
    ax.set_yscale('log')
    ax.set_xlabel('Stage / Anomaly Level')
    ax.set_ylabel('Anomaly Index')
    
    # Add UCL
#    ax.axhline(UCL, color='k', linestyle='-.', linewidth=2, label='UCL')
#    ax.text(8, UCL *1.5, 'UCL', rotation=0, va='center')
    if system_name:
        ax.set_title(f'System {system_name}')
    
    ax.grid(True, axis='y', which='minor', linestyle='--', linewidth=0.5)
    return ax

@dataclass
class Benchmark_SA:
    ad_system: AD_system
    dl : Union[Dataset,DataLoader]
    batch_size: int = None

    def __post_init__(self):
        
        if isinstance(self.dl,Dataset):
            assert self.batch_size is not None, 'batch_size must be provided'
            self.dl = DataLoader(self.dl, batch_size=self.batch_size, shuffle=False)
        
        
        self.df_res = compute_anomaly_index(self.dl,self.ad_system)

    def Compute_anomaly_index(self):
        return compute_anomaly_index(self.dl,self.ad_system)
    
    def evaluate_one_system(self,system_name:int,window:int=None):
        df_sys = select_system(self.df_res,system_name)
        df_sys = order_df(df_sys)
        df_sys = scale_anomaly_index(df_sys)
        auc = get_auc_per_level(df_sys)
        if window is not None:
            df_sys = average_window(df_sys,window)
        UCL, LCL = get_CL(df_sys)
        ax = plot_boxplot_with_train(df_sys, UCL,system_name=system_name)
        return auc, ax
    
    def evaluate_all_systems(self,window:int=None):
        systems = self.df_res['system_name'].unique()
        auc_dict=defaultdict(lambda: defaultdict(dict))
        axs = dict()
        for system in systems:
            system_name = f'system_{system}'
            auc, ax = self.evaluate_one_system(system,window)
            auc_dict[system_name] = auc
            axs[system_name] = ax

        return auc_dict, axs
    
if __name__ == '__main__':
    from pathlib import Path
    from config import settings, create_psd_path
    from psm.models.vanilla_classification import DenseSignalClassifierModule
    from psm.models.ad_systems import AD_GMM
    from psm.models.prepare_data import CreateTransformer, PSDDataModule, PSDDataset, PSDDataset_test
    from psm.utils.data.metadata import get_metadata_processed

    settings_proc = 'SETTINGS1'
    settings_simu = 'SETTINGS1'
    root = Path(settings.data.path["processed"])

    database_path = create_psd_path(root,settings_proc,settings_simu)
    print(database_path.exists())
    model_paths = Path(settings.data.path['model']) / 'model'
    name_model_1 = 'best-epoch=28-val_loss=2.13.ckpt'

    model = DenseSignalClassifierModule.load_from_checkpoint(model_paths / name_model_1)
    ad_gmm = AD_GMM(num_classes=20, model=model.model)


    metadata = get_metadata_processed(settings_proc, settings_simu)
    freq_axis = metadata['freq']

    transformer = CreateTransformer(database_path, freq=freq_axis, freq_min=0, freq_max=150)
    transform_psd = transformer.transform_psd
    transform_label = transformer.transform_label
    dm = PSDDataModule(database_path, transform_psd, transform_label, batch_size=32)
    dm.setup()
    train_dl = dm.train_dataloader()
    ad_gmm.fit(train_dl)
    psd_test = PSDDataset_test(database_path=database_path, transform=transform_psd, transform_label=transform_label)
    psd_dl = DataLoader(psd_test, batch_size=10000)

    benchmark_sa = Benchmark_SA(ad_gmm,psd_dl)

    auc, axs = benchmark_sa.evaluate_all_systems()
    print(auc['system_1'])


