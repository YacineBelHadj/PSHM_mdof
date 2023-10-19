from torch.utils.data import DataLoader, Dataset
from typing import Union
import pandas as pd


from functools import partial
import numpy as np
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from tqdm import tqdm

from psm.eval.aucs_computation import compute_auc_for_levels , compute_auc
from psm.utils.data.resonance_frequnecy import get_res_freq
class Benchmark_SA():
    def __init__(self,ad_system,anomaly_dl:Union[Dataset,DataLoader],
                 test_dl:Union[Dataset,DataLoader], batch_size=None):
        
        self.batch_size = batch_size
        self.ad_system = ad_system
        # check if the model has predict and fit method
        if not hasattr(self.ad_system, 'predict'):
            raise ValueError("The model does not have predict method")
        if not hasattr(self.ad_system, 'fit'):
            raise ValueError("The model does not have fit method")
  
         
        if isinstance(anomaly_dl,Dataset):
            anomaly_dl = DataLoader(anomaly_dl, batch_size=self.batch_size, shuffle=False)
        if isinstance(test_dl,Dataset):
            test_dl = DataLoader(test_dl, batch_size=self.batch_size, shuffle=False)

        self.anomaly_dl = anomaly_dl
        self.test_dl = test_dl

    
    def evaluate(self):

        system_names, anomaly_levels, log_likelihoods = [], [], []
        for psd, system_name, anomaly_level in tqdm(self.anomaly_dl,
                                                    desc = 'benchmarking_sa',
                                                    total=len(self.anomaly_dl)):
            log_likelihood = self.ad_system.predict(psd)
            system_names.extend(system_name.tolist())
            anomaly_levels.extend(anomaly_level.tolist())
            
            log_likelihoods.extend(log_likelihood)
        
        for psd, system_name in tqdm(self.test_dl,
                                     desc = 'benchmarking_sa',
                                     total=len(self.test_dl)):
            log_likelihood = self.ad_system.predict(psd)
            system_names.extend(system_name.tolist())
            anomaly_levels.extend([0]*len(system_name))
            log_likelihoods.extend(log_likelihood)
        
        df_res = pd.DataFrame({
            'system_name': system_names,
            'anomaly_level': anomaly_levels,
            'log_likelihood': log_likelihoods
        })
        result = compute_auc_for_levels(df_res)
        return result
 ##########
import pandas as pd 
import matplotlib.pyplot as plt
from functools import partial
from scipy.integrate import simps as simpson
from torch.utils.data import DataLoader, Dataset
from psm.utils.data.resonance_frequnecy import get_res_freq

# Function Definitions
def compute_auc_scores(df_notched_resonance, system_number=0):
    df_system = df_notched_resonance[df_notched_resonance['system_name']==system_number]
    df_healthy = df_system[(df_system['amplitude']==0) & (df_system['f_notch']==0)]
    compute_auc_partial = partial(compute_auc, healthy=df_healthy['log_likelihood'].values)
    df_grouped = df_system.groupby(['f_notch','amplitude'])['log_likelihood'].apply(compute_auc_partial)
    df_grouped = df_grouped.reset_index()
    df_grouped = df_grouped[df_grouped["f_notch"]!=0]
    df_grouped['AUC'] = np.abs(0.5-df_grouped['log_likelihood'])+0.5
    df_grouped.drop(columns='log_likelihood', inplace=True)
    return df_grouped

def compute_total_area(contour):
    contour_paths = contour.collections[1].get_paths()
    x, y = [], []
    for contour_path in contour_paths:
        for vertex in contour_path.to_polygons():
            x.extend(vertex[:,0])
            y.extend(vertex[:,1])
    x = np.array(x)
    y = np.array(y)
    x1, y1 = x[y>=0], y[y>=0]
    x2, y2 = x[y<0], y[y<0]
    x1, y1 = zip(*sorted(zip(x1, y1)))
    x2, y2 = zip(*sorted(zip(x2, y2)))
    area1 = np.abs(simpson(y1, x1))
    area2 = np.abs(simpson(y2, x2))
    area_tot = area1+ area2
    return  area_tot

def generate_heatmap_data(df_grouped_auc):
    df_grouped_auc['AUC'] = np.abs(0.5-df_grouped_auc['AUC'])+0.5
    heatmap_data = df_grouped_auc.pivot(index='f_notch', columns='amplitude', values='AUC')
    heatmap_data = heatmap_data.transpose()
    return heatmap_data 

def plot_auc_contour(heatmap_data, system_name):
    resonance_freq = get_res_freq()
    resonance_freq_interest = resonance_freq.loc[system_name]
    contour_levels = np.arange(0.5, 1, 0.1)
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contour(
        heatmap_data.columns,
        heatmap_data.index,
        heatmap_data.values,
        levels=contour_levels,
        cmap='seismic'
    )
    for res in resonance_freq_interest.values[:-1]:
        plt.axvline(res, color='black', linestyle='--')
    ax.set_xlabel('f_notch')
    ax.set_ylabel('amplitude')
    plt.colorbar(contour)  
    ax.set_title(f'Contour plot of AUC for {system_name}')
    return ax, contour

def compute_auc_heatmap_weighted(heatmap_data):
    heatmap_data = heatmap_data[heatmap_data.index!=0]
    amplitudes = np.abs(heatmap_data.index).values
    inverse_amplitude = (1/(amplitudes**2)).flatten()
    aucs = heatmap_data.values
    weighted_auc = inverse_amplitude@aucs
    weighted_auc = np.sum(weighted_auc)/ np.sum(inverse_amplitude*heatmap_data.shape[1])
    return {'weighted_auc_VAS':weighted_auc}

def compute_auc_heatmap_metrics(heatmap_data):
    heatmap_data = heatmap_data[heatmap_data.index!=0]

    flat_df  = heatmap_data.reset_index().values.flatten()
    harmonic_mean = len(flat_df)/np.sum(1/flat_df)
    mean = np.mean(flat_df)
    geometric_mean = np.prod(flat_df)**(1/len(flat_df))
    return {"VAS_hm":harmonic_mean, "VAS_m":mean, "VAS_gm":geometric_mean}

# Class Definitions
class Benchmark_VAS:
    def __init__(self, anomaly_detector, notched_psd_dataloader, original_psd_dataloader, batch_size=None):
        self.anomaly_detector = anomaly_detector
        self.notched_psd_dataloader = DataLoader(notched_psd_dataloader, batch_size=batch_size, shuffle=False)\
              if isinstance(notched_psd_dataloader, Dataset) else notched_psd_dataloader
        
        self.original_psd_dataloader = DataLoader(original_psd_dataloader, batch_size=batch_size, shuffle=False)\
              if isinstance(original_psd_dataloader, Dataset) else original_psd_dataloader
        
            
    def evaluate_notched_systems(self):
        systems_names, f_notch, amplitude, log_likelihoods = [], [], [], [] 

        for dataLoader in [self.original_psd_dataloader, self.notched_psd_dataloader]:
            for psd, sys_name, amplitude_notch,f_affect in tqdm(dataLoader,
                                                                desc = 'benchmarking_vas',
                                                                total=len(dataLoader)):
                log_likelihood = self.anomaly_detector.predict(psd)
                systems_names.extend(sys_name.tolist())
                log_likelihoods.extend(log_likelihood.tolist())
                f_notch.extend(f_affect.tolist())
                amplitude.extend(amplitude_notch.tolist())
        df_notched_resonance = pd.DataFrame({
            'system_name': systems_names,
            'f_notch': f_notch,
            'amplitude': amplitude,
            'log_likelihood': log_likelihoods})
        return df_notched_resonance   
    
    def evaluate_single_system(self, df_notched_resonance, system_number):
        system_name = f'system_{system_number}'
        df_grouped_auc = compute_auc_scores(df_notched_resonance, system_number=system_number)
        heatmap_data = generate_heatmap_data(df_grouped_auc)
        ax, contour = plot_auc_contour(heatmap_data, system_name)
        #area = compute_total_area(contour)
        metrics = compute_auc_heatmap_metrics(heatmap_data)
        weighted_auc = compute_auc_heatmap_weighted(heatmap_data)
        metrics.update(weighted_auc)
        return  ax, metrics 
    
    def evaluate_all_systems(self):
        df_notched_resonance = self.evaluate_notched_systems()
        system_numbers = df_notched_resonance['system_name'].unique()
        axs, metrics_list = dict(), dict()
        for system_number in system_numbers:
            ax, metrics = self.evaluate_single_system(df_notched_resonance, system_number)
            
            #areas[f'system_{system_number}'] = area
            axs[f'system_{system_number}'] = ax
            metrics_list[f'system_{system_number}'] = metrics

        #mean_areas = np.mean(list(areas.values()))

        res={'axs':axs, 'individual_metric':metrics_list}
        return res