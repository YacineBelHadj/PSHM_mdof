from torch.utils.data import DataLoader, Dataset
from typing import Union
import pandas as pd


from functools import partial
import numpy as np
from scipy.integrate import simpson
import matplotlib.pyplot as plt

from psm.eval.aucs_computation import compute_auc_for_levels , compute_auc
from psm.utils.data.resonance_frequnecy import get_res_freq
class Benchmark_SA():
    def __init__(self,ad_system):
        self.ad_system = ad_system
        # check if the model has predict and fit method
        if not hasattr(self.ad_system, 'predict'):
            raise ValueError("The model does not have predict method")
        if not hasattr(self.ad_system, 'fit'):
            raise ValueError("The model does not have fit method")
    
    def evaluate(self, anomaly_dl:Union[Dataset,DataLoader],
                  test_dl:Union[Dataset,DataLoader], 
                  batch_size:int):
        if isinstance(anomaly_dl,Dataset):
            anomaly_dl = DataLoader(anomaly_dl, batch_size=batch_size, shuffle=False)
        if isinstance(test_dl,Dataset):
            test_dl = DataLoader(test_dl, batch_size=batch_size, shuffle=False)

        system_names, anomaly_levels, log_likelihoods = [], [], []
        for psd, system_name, anomaly_level in anomaly_dl:
            log_likelihood = self.ad_system.predict(psd)
            system_names.extend(system_name.tolist())
            anomaly_levels.extend(anomaly_level.tolist())
            log_likelihoods.extend(log_likelihood)
        
        for psd, system_name in test_dl:
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
        auc_03 = result[0.03]
        mean_auc_03 = auc_03.mean()

        auc_03 = {f"auc_03_{k}":v for k,v in auc_03.items()}


        return result, auc_03, mean_auc_03
 ##########
def compute_aucs(df_res_notch,system_num=0):
        df_sys = df_res_notch[df_res_notch['system_name']==system_num]
        df_health = df_sys[(df_sys['amplitude']==0) & (df_sys['f_notch']==0)]
        compute_auc_partial = partial(compute_auc, healthy=df_health['log_likelihood'].values)
        
        df_grouped = df_sys.groupby(['f_notch','amplitude'])['log_likelihood'].apply(compute_auc_partial)
        df_grouped = df_grouped.reset_index()
        df_grouped = df_grouped[df_grouped["f_notch"]!=0]
        df_grouped['AUC'] = np.abs(0.5-df_grouped['log_likelihood'])+0.5
        df_grouped.drop(columns='log_likelihood', inplace=True)
        return df_grouped

def calculate_area(contour):
        contour_paths = contour.collections[1].get_paths()
        x,y = [],[]
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

def plot_contour(df_grouped,sys_name):
        resonance_freq=get_res_freq()
        res_freq_interesset = resonance_freq.loc[sys_name]
        df_grouped['AUC']= np.abs(0.5-df_grouped['AUC'])+0.5
        heatmap_data = df_grouped.pivot(index='f_notch', columns='amplitude', values='AUC')
        heatmap_data = heatmap_data.transpose()

        contour_levels = np.arange(0.5, 1, 0.1)
        fig,ax = plt.subplots(figsize=(10, 8))
        contour = ax.contour(
            heatmap_data.columns,
            heatmap_data.index,
            heatmap_data.values,
            levels=contour_levels,
            cmap='seismic'
        )
        for res in res_freq_interesset.values[:-1]:
            plt.axvline(res, color='black', linestyle='--')
        ax.set_xlabel('f_notch')
        ax.set_ylabel('amplitude')
        plt.colorbar(contour)  
        ax.set_title(f'Contour plot of AUC for {sys_name}')
        return ax , contour


class Benchmark_VAS:
    def __init__(self,ad_system,psd_notch_dl,psd_notch_original_dl,batch_size=None):
        self.ad_system = ad_system
        if not hasattr(self.ad_system, 'predict'):
            raise ValueError("The model does not have predict method")
        
        self.psd_notch_dl = psd_notch_dl
        if isinstance(self.psd_notch_dl,Dataset):
            self.psd_notch_dl = DataLoader(self.psd_notch_dl, 
                                           batch_size=batch_size, shuffle=False)

        self.psd_notch_original_dl = psd_notch_original_dl
        if isinstance(self.psd_notch_original_dl,Dataset):
            self.psd_notch_original_dl = DataLoader(self.psd_notch_original_dl, 
                                                    batch_size=batch_size, shuffle=False)
            
    def evaluate_on_notch(self):
        systems_names, f_notch, amplitude, log_likelihoods = [], [], [], [] 
        dataLoader_list=[self.psd_notch_original_dl,self.psd_notch_dl]
       
        for dataLoader in dataLoader_list:
            for psd, sys_name, amplitude_notch,f_affect in dataLoader:
                log_likelihood = self.ad_system.predict(psd)
                systems_names.extend(sys_name.tolist())
                log_likelihoods.extend(log_likelihood.tolist())
                f_notch.extend(f_affect.tolist())
                amplitude.extend(amplitude_notch.tolist())

        df_res_notch = pd.DataFrame({
            'system_name': systems_names,
            'f_notch': f_notch,
            'amplitude': amplitude,
            'log_likelihood': log_likelihoods})
        return df_res_notch     
    
    def evaluate_system(self,df_res_notch,system_num):
        system_name = f'system_{system_num}'
        df_grouped = compute_aucs(df_res_notch,system_num)
        ax,contour = plot_contour(df_grouped,system_name)
        area = calculate_area(contour)
        return area, ax
    
    def evaluate_all_systems(self):
        df_res_notch = self.evaluate_on_notch()
        systems_names = df_res_notch['system_name'].unique()
        areas = {}

        axs = {}
        for sys_name in systems_names:
            area,ax = self.evaluate_system(df_res_notch,sys_name)
            areas['area_vas_'+str(sys_name)] = area
            axs['system_'+str(sys_name)] = ax
        mean_areas = np.mean(list(areas.values()))
        return areas, mean_areas, axs
    
    


