#%% 
import logging
from tqdm import tqdm
import pandas as pd
from functools import partial
from pselia.eval.aucs_computation import compute_auc
import numpy as np
from scipy import stats as st
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Union, List, ClassVar , Dict
from pselia.training.ad_systems import AD_system , AD_GMM
from torch.utils.data import DataLoader, Dataset
# garbagge collector
import gc

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

def compute_anomaly_score(model, dataloaders):
    assert isinstance(dataloaders, list), 'dataloaders should be a list of dataloaders'
    sensor_face, sensor_dir ,f_notch_list,amplitude_notch, anomaly_index = [],[],[],[],[]
    for dataloader in dataloaders:
        for batch in tqdm(dataloader, total=len(dataloader)): 
            
            psd, face, direction, f_affected, amplitude, datetime = batch 
            try :
                output = model.predict(psd)
            except ValueError as e: 
                inf_index = np.where(np.isinf(psd.max(axis=1)))[0]
                # datetime is tuple let's read the inf_index
                datetime = [datetime[i] for i in inf_index]
                print(datetime)

                plt.figure()
                plt.plot(psd.T)
                plt.show()
                plt.close()
                raise e
            sensor_dir.extend(direction)
            sensor_face.extend(face)
            f_notch_list.extend(f_affected.numpy())
            amplitude_notch.extend(amplitude.numpy())  # Assuming amplitude is also array-like
            anomaly_index.extend(output)
                    
    df_res = pd.DataFrame({'face':sensor_face,
                            'direction':sensor_dir,
                            'aff_f':f_notch_list,
                            'aff_amp':amplitude_notch,
                            'anomaly_index':anomaly_index})
    return df_res

def scale_anomaly_index(df_sensor):
    df_sensor.loc[:,'anomaly_index'] = -1*df_sensor['anomaly_index']
    df_sensor.loc[:,'anomaly_index'] = df_sensor['anomaly_index'] - df_sensor['anomaly_index'].min()+1
    return df_sensor

def select_sensor(df,face:str,direction:str):
    df = df[(df['face']==face) & (df['direction']==direction)]
    return df

def add_auc_col(df_sensor):
    df_healthy = df_sensor[(df_sensor['aff_f']==0) & (df_sensor['aff_amp']==0)]
    assert len(df_healthy)>0, 'healthy data should be present'

    compute_auc_partial = partial(compute_auc,healthy=df_healthy['anomaly_index'].values)
    # remove the healthy data
    df_sensor = df_sensor[(df_sensor['aff_f']!=0) | (df_sensor['aff_amp']!=0)]
    df_grouped = df_sensor.groupby(['aff_f','aff_amp'])['anomaly_index'].apply(compute_auc_partial)
    df_grouped=df_grouped.reset_index()
    df_grouped['AUC']=df_grouped['anomaly_index']
    # assert that the AUC are between 0 and 1
    if not df_grouped['AUC'].between(0.1,1).all():
        print('AUC should be between 0 and 1')
    return df_grouped

def compute_means(df_aucs):
    if 'AUC' not in df_aucs.columns:
        raise ValueError('AUC should be a column name')
    df_auc = df_aucs[df_aucs['aff_amp']!=0]['AUC'] # non 0 amplitude
    ## flatten all the values
    auc_array = df_auc.values.flatten()
    ## saturate the values between 0.5 and 1 this is needed for very bad models
    ## boolean variable to check is we need to saturate the values
    saturate = (auc_array<0.3).any()
    auc_array = np.clip(auc_array,0.3,1)

    harmonic_mean = st.hmean(auc_array)
    arithmetic_mean = np.mean(auc_array)
    return {'harmonic_mean':harmonic_mean,
            'arithmetic_mean':arithmetic_mean,
            'saturation':saturate}

def plot_contour(df_aucs,ax,face:str,direction:str):
    contour_data = df_aucs.pivot(index='aff_amp',columns='aff_f',values='AUC')
    X = np.array(contour_data.columns)
    Y = np.array(contour_data.index)
    Z = contour_data.values
    contour_levels = [0.55,0.6,0.7,0.8,0.9,1.0]

    cp = ax.contour(X,Y,Z,cmap ='viridis',levels=contour_levels)
    cbar = plt.colorbar(cp)
    cbar.set_label('AUC',fontsize=14)
    ax.set_title(f'Anomaly index for {direction} direction and face {face}')
    ax.set_xlabel('Amplitude')
    ax.set_ylabel('Frequency (aff_freq)')
    ax.grid(True, linestyle='--', linewidth=0.5, color='grey')

    return ax

@dataclass
class Benchmark_VAS:
    anomaly_detector: AD_system
    notched_psd_dataloader:  DataLoader = None
    original_psd_dataloader: DataLoader = None
    
    def __post_init__(self):
        self.df_res = self.Compute_anomaly_score()
    
    def Compute_anomaly_score(self):
        return compute_anomaly_score(self.anomaly_detector,
                                    [self.notched_psd_dataloader,
                                    self.original_psd_dataloader])
    
    def evaluate_one_sensor(self,face,direction):
        df_sys = select_sensor(self.df_res,face,direction)
        df_sys_processed = scale_anomaly_index(df_sys)
        df_sys_processed = add_auc_col(df_sys_processed)
        metrics = compute_means(df_sys_processed)
        fig,ax = plt.subplots(figsize=(10,8))
        ax = plot_contour(df_sys_processed,ax,face,direction)    
        plt.close(fig)
        gc.collect()
        return metrics, ax
    
    def evaluate_all_sensor(self):
        faces = self.df_res['face'].unique()
        directions = self.df_res['direction'].unique()
        metrics = {}
        axs = {}
        for face in faces:
            for direction in directions:
                metrics[(face,direction)], axs[(face,direction)] = self.evaluate_one_sensor(face,direction)
        gc.collect()
        return metrics, axs
#%%
if __name__ =='__main__':
    #%%
    from pselia.training.datamodule import CreateTransformer, PSDNotchDataset, PSDELiaDataModule
    from pselia.training.dense_model import DenseSignalClassifierModule
    from pselia.config_elia import settings, load_processed_data_path_vas, load_processed_data_path
    from pselia.utils import load_freq_axis
    #%%
    logging.info('Start')
    settings_proc = 'SETTINGS1'
    vas_path = load_processed_data_path_vas(settings_proc)    
    database_path = load_processed_data_path(settings_proc)
    database_path = load_processed_data_path('SETTINGS1')
    freq_axis = load_freq_axis(database_path)
    freq_min , freq_max = settings.neuralnetwork.settings1.freq_range

    transformer = CreateTransformer(database_path, freq_axis, freq_min=freq_min, freq_max=freq_max)
    transform_psd = transformer.transform_psd
    transform_label = transformer.transform_label
    input_dim = transformer.dimension_psd()

    dm = PSDELiaDataModule(database_path=database_path, 
                        transform=transform_psd, label_transform=None, 
                        batch_size=32)

    dm.setup()
    dl_feature= dm.ad_system_dataloader(batch_size=30000)
    logging.info('dl_feature loaded ')
    ds_notch = PSDNotchDataset(database_path=vas_path, 
                            transform=transform_psd, label_transform=None)
    ds_original = PSDNotchDataset(database_path=vas_path, 
                                transform=transform_psd, label_transform=None, original_psd=True)
    dl_notch = DataLoader(ds_notch, batch_size=10000, shuffle=False, num_workers=2)   
    dl_original = DataLoader(ds_original, batch_size=10000, shuffle=False, num_workers=2)
    model_paths = '/home/yacine/Documents/PhD/Code/GitProject/PBSHM_mdof/model/model_elia/best-epoch=99-val_loss=0.00.ckpt'
    model =DenseSignalClassifierModule .load_from_checkpoint(model_paths)
    ad_system = AD_GMM(num_classes=12, model=model.model)
    ad_system.fit(dl_feature)
    logging.info('AD system fitted')
    #%%
    benchmark = Benchmark_VAS(ad_system,dl_notch,dl_original)
    logging.info('Benchmark done')
    #%%
    metrics, axs = benchmark.evaluate_one_sensor(face='2',direction='X') 
    fig=axs.get_figure()   
    plt.show(fig)
    plt.close()
    #plot boxplot for all sensors
# %%
