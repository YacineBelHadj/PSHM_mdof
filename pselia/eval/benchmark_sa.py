from pselia.training.datamodule import CreateTransformer, PSDELiaDataset_test, PSDELiaDataModule
from pselia.training.dense_model import DenseSignalClassifierModule
from pselia.config_elia import settings, load_processed_data_path
from pselia.utils import load_freq_axis, get_event_to_abbr
from pselia.training.ad_systems import AD_GMM
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from pselia.eval.aucs_computation import compute_auc
import numpy as np
from functools import partial
from pselia.training.ad_systems import AD_system
from dataclasses import dataclass
from typing import Union, List, ClassVar , Dict
from collections import defaultdict
import pandas as pd 
import gc
import matplotlib.pyplot as plt

def select_sensor(df,face:str,direction:str):
    df = df.loc[(df['face']==face) & (df['direction']==direction)]
    return df

def scale_anomaly_index(df_sys):
    df_sys.loc[:,'anomaly_index'] = -1*df_sys['anomaly_index']
    df_sys.loc[:,'anomaly_index'] = df_sys['anomaly_index'] - df_sys['anomaly_index'].min()+1
    return df_sys

def get_auc_per_anomaly(df_sys):
    test_data = df_sys.loc[df_sys['stage']=='testing']
    anoumalous_data = df_sys.loc[df_sys['stage']=='anomaly']
    compute_auc_partial = partial(compute_auc,healthy=test_data['anomaly_index'].values)
    df_grouped = anoumalous_data.groupby(['anomaly_description'])['anomaly_index'].apply(compute_auc_partial)
    return df_grouped

def average_window(df_sys, window=20):
    # Apply rolling median only within each 'stage' and 'anomaly_level' group
    df_sys.loc[:,'anomaly_index'] = df_sys['anomaly_index'].transform(lambda x: x.rolling(window).median())
    return df_sys

def compute_anomaly_index(psd_dl:DataLoader,
                          ad_system:AD_system):
    lists = {
        'anomaly_index': [],
        'anomaly_description': [],
        'stage': [],
        'face': [],
        'direction': [],
        'date_time': [],
    }
    for batch in psd_dl:
        psd, face, direction, date_time, stage, anomaly_description = batch
        anomaly_idx = ad_system.predict(psd)
        # Convert anomaly_idx to a tuple
        anomaly_idx = tuple(anomaly_idx)
        # Append values to lists
        for key, value in zip(lists.keys(), [anomaly_idx, anomaly_description, stage, face, direction, date_time]):
            lists[key].extend(value)

    
    assert isinstance(face[0],str), 'face variable must be a string please dont incluse transformation in the dataloader'
    assert isinstance(direction[0],str), 'direction must be a string please dont incluse transformation in the dataloader'
    
    df_res = pd.DataFrame(lists)
    df_res.loc[:,'date_time'] = pd.to_datetime(df_res['date_time'])
    df_res.set_index('date_time',inplace=True)
    return df_res

import seaborn as sns


def plot_boxplot(df_sys_processed, direction, face, ordered_categories, auc):
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create the boxplot
    box_plot = sns.boxplot(x='hue_plot', y='anomaly_index', data=df_sys_processed, ax=ax,
                           order=ordered_categories, showfliers=False, color='lightblue')

    # Annotate each box with the corresponding AUC value
     # Iterate over the AUC dictionary


    # Set plot title and labels
    ax.set_title(f'Anomaly Index for {direction.capitalize()} Direction and Face {face}', fontsize=14, weight='bold')
    ax.set_xlabel('Anomaly Description', fontsize=12)
    ax.set_ylabel('Anomaly Index', fontsize=12)

    plt.xticks(rotation=45, ha="right", rotation_mode="anchor", fontsize=10)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.subplots_adjust(bottom=0.15)
    plt.close()
    return ax




@dataclass
class Benchmark_SA:
    mapping_event_to_abbr: ClassVar[Dict[str, str]] = get_event_to_abbr()

    ad_system:AD_system
    dl : Union[Dataset,DataLoader]
    batch_size:int = 2000

    def __post_init__(self):
        if isinstance(self.dl, Dataset):
            assert self.batch_size is not None, 'batch_size must be specified'
            self.dl = DataLoader(self.dl, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.df_res = compute_anomaly_index(self.dl, self.ad_system)
    
    def Compute_anomaly_index(self):
        return compute_anomaly_index(self.df, self.ad_system)

    def add_abbreviation_hue(self,df_sys):
        df_sys = df_sys.copy()
        df_sys.loc[:, 'hue_plot'] = df_sys['anomaly_description']
        df_sys.loc[:,'hue_plot'] = df_sys['anomaly_description']
        df_sys.loc[df_sys['stage']!='anomaly','hue_plot'] = df_sys['stage']    
        # sort the data by the index that is date_time
        df_sys = df_sys.sort_values(by=['date_time'])
        earliest_date_mapping = df_sys.reset_index().groupby('hue_plot')['date_time'].min().sort_values()

        # Reorder the 'hue_plot' categories based on the earliest_date_mapping
        ordered_categories = earliest_date_mapping.index
        df_sys.loc[:,'hue_plot'] = pd.Categorical(df_sys['hue_plot'], categories=ordered_categories, ordered=True)
        return df_sys , earliest_date_mapping
    
    def evaluate_one_sensor(self,face,direction,window:int=None):
        df_sys = select_sensor(self.df_res,face,direction)
        df_sys_processed = scale_anomaly_index(df_sys)
        df_sys_processed.sort_index(inplace=True)
        auc = get_auc_per_anomaly(df_sys_processed)
    
        if window is not None:
            df_sys_processed = average_window(df_sys_processed,window)
        df_sys_processed, earliest_date_maping  = self.add_abbreviation_hue(df_sys_processed)
        ax = plot_boxplot(df_sys_processed,direction,face,ordered_categories=earliest_date_maping.index, auc=auc)
        gc.collect()
        return auc, ax, earliest_date_maping
    
    def evaluate_all_systems(self, window:int=None):
        faces = self.df_res['face'].unique()
        directions = self.df_res['direction'].unique()
        auc_dict = defaultdict(lambda : defaultdict(dict))
        axs = dict()
        for face in faces:
            for direction in directions:
                sensor = (face,direction)
                sensor_name = f'face {sensor[0]} direction {sensor[1]}'
                auc, ax , ea= self.evaluate_one_sensor(*sensor,window=window)
                sorted_auc_data = auc.reindex(ea.index.intersection(auc.index))
                auc_dict[sensor] = sorted_auc_data

                axs[sensor_name] = ax
        gc.collect()
        return auc_dict, axs



if __name__=='__main__':
    database_path = load_processed_data_path('SETTINGS1')
    freq_axis = load_freq_axis(database_path)
    freq_min , freq_max = settings.neuralnetwork.settings1.freq_range

    transformer = CreateTransformer(database_path, freq_axis, freq_min=freq_min, freq_max=freq_max)
    transform_psd = transformer.transform_psd
    transform_label = transformer.transform_label
    input_dim = transformer.dimension_psd()

    ### data loader
    dm = PSDELiaDataModule(database_path=database_path, 
                        transform=transform_psd, label_transform=transform_label, 
                        batch_size=32)
    dm.setup()
    dl_feature= dm.ad_system_dataloader(batch_size=2000)
    ds = PSDELiaDataset_test(database_path=database_path, 
                            transform=transform_psd, label_transform=None)
    dl_all = DataLoader(ds, batch_size=2000, shuffle=False, num_workers=4)


    ## loader model 
    model_paths = '/home/yacine/Documents/PhD/Code/GitProject/PBSHM_mdof/model/model_elia/best-epoch=99-val_loss=0.00.ckpt'
    model = DenseSignalClassifierModule.load_from_checkpoint(model_paths)
    ad_system = AD_GMM(num_classes=12, model=model.model)
    ad_system.fit(dl_feature)
    benchmark = Benchmark_SA(ad_system,dl_all)
    auc_dict, axs = benchmark.evaluate_all_systems(window=20)
    #plot boxplot for all sensors
    for ax in axs.values():
        ax.get_figure()
        plt.show()
        plt.close()
        break