#%%

# let's now build a class that does the benchmarking for us 
# and return the contour plot  and the harmonic mean 
# for tha given model , and all systems
from typing import Union, List, Dict, Tuple
from psm.models.ad_systems import AD_system
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset
from functools import partial
import numpy as np
from psm.eval.aucs_computation import compute_auc
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from scipy import stats as st
# logging
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def compute_anomaly_score(model,dataloaders):
    if not isinstance(dataloaders,list):
        raise ValueError('dataloaders should be a list of dataloaders')

    sys_names_list, f_notchs_list, amplitudes_list, log_likelihoods_list = [], [], [], []
    for dl in dataloaders:
        for psd,sys_name,amplitude_notch,f_affect in tqdm(dl,total=len(dl)):
            output = model.predict(psd)
            log_likelihoods_list.extend(output.tolist())
            sys_names_list.extend(sys_name.tolist())
            f_notchs_list.extend(f_affect.tolist())
            amplitudes_list.extend(amplitude_notch.tolist())
        
    df_res = pd.DataFrame({'system_name':sys_names_list,
                            'f_notch':f_notchs_list,
                            'amplitude':amplitudes_list,
                            'anomaly_index':log_likelihoods_list})
    return df_res

def add_auc_col(df_res,system_id:int=1):
    df_sys = df_res[df_res['system_name']==system_id]
    df_sys.loc[:,'anomaly_index'] = -df_sys['anomaly_index']
    df_healthy = df_sys[(df_sys['f_notch']==0) & (df_sys['amplitude']==0)]
    compute_auc_partial = partial(compute_auc,healthy=df_healthy['anomaly_index'].values)
    df_grouped = df_sys.groupby(['f_notch','amplitude'])['anomaly_index'].apply(compute_auc_partial)
    df_grouped=df_grouped.reset_index()
    df_grouped['AUC']=df_grouped['anomaly_index']
    # assert that the AUC are between 0 and 1
    if not df_grouped['AUC'].between(0.1,1).all():
        print('AUC should be between 0 and 1')

    return df_grouped   

def compute_means(df_aucs):
    # remove 0 amplitude
    # check if AUC is column name
    if 'AUC' not in df_aucs.columns:
        raise ValueError('AUC should be a column name')
    
    df_auc = df_aucs[df_aucs['amplitude']!=0]['AUC'] # non 0 amplitude
    ## flatten all the values
    auc_array = df_auc.values.flatten()
    ## saturate the values between 0.5 and 1 this is needed for very bad models 
    ## boolean variable to check is we need to saturate the values
    saturate = (auc_array<0.3).any()
    auc_array = np.clip(auc_array,0.1,1)
    # compute the geometric mean
    geometric_mean = st.gmean(auc_array)
    # compute the harmonic mean
    harmonic_mean = st.hmean(auc_array)
    # compute the mean - arthmetic mean
    mean = np.mean(auc_array)

    return {'geometric_mean':geometric_mean, 'harmonic_mean':harmonic_mean, 'arthmetic_mean':mean,
            'saturation':saturate}



def plot_contour(df_aucs, ax, system_id=None, df_resonance_avg=None):
    # Prepare data for contour plot
    contour_data = df_aucs.pivot(index='amplitude', columns='f_notch', values='AUC')
    X = np.array(contour_data.columns)
    Y = np.array(contour_data.index)
    Z = contour_data.values
    contour_levels = [0.6, 0.7, 0.8, 0.9, 1.0]

    # Plot contour
    cp = ax.contour(X, Y, Z, cmap='viridis', levels=contour_levels)
    cbar = plt.colorbar(cp, ax=ax, orientation='vertical')
    cbar.set_label('AUC', fontsize=14)

    ax.set_xlabel('Frequency (f_notch)', fontsize=14)
    ax.set_ylabel('Amplitude', fontsize=14)
    ax.grid(True, linestyle='--', linewidth=0.5, color='grey')
    
    if system_id and df_resonance_avg is not None:
        res_freq = df_resonance_avg[df_resonance_avg['system_id']==system_id].values.flatten()[1:]
        for fi in res_freq[1:-1]:
            ax.axvline(fi, color='black', linestyle='-.')
    # adding tilte system id
    if system_id:
        ax.set_title(f'System {system_id}')
    ax.set_yticks([i/10 for i in range(-5,6)])
    ax.set_xticks([i for i in range(0,150,20)])
    return ax


@dataclass
class Benchmark_VAS:
    """ Class for evaluating an anomaly detection system
    on virtually synthesized data. """
    anomaly_detector: AD_system
    notched_psd_dataloader: Union[DataLoader, List[DataLoader]]
    original_psd_dataloader: Union[DataLoader, List[DataLoader]]
    batch_size: int = None
    df_resonance_avg: pd.DataFrame = None
    
    def __post_init__(self):

        if not isinstance(self.notched_psd_dataloader, DataLoader):
            self.notched_psd_dataloader = DataLoader(self.notched_psd_dataloader, 
                                                batch_size=self.batch_size, shuffle=False)
            
        if not isinstance(self.original_psd_dataloader, DataLoader):
            self.original_psd_dataloader = DataLoader(self.original_psd_dataloader, 
                                                 batch_size=self.batch_size, shuffle=False)
        
        self.df_res = self.Compute_anomaly_score()
    
    def Compute_anomaly_score(self):
        df_res = compute_anomaly_score(self.anomaly_detector,
                              [self.original_psd_dataloader, self.notched_psd_dataloader])
        return df_res
    
    
    def evaluate_one_individu(self,system_id:int):
        df_sys = add_auc_col(self.df_res,system_id=system_id)
        optimization_metrics = compute_means(df_sys)
        fig,ax= plt.subplots(figsize=(10,8))
        ax = plot_contour(df_sys, ax, system_id=system_id, df_resonance_avg=self.df_resonance_avg)
        plt.close(fig)
        return optimization_metrics, ax
    
    def evaluate_all_individus(self):
        id_list = self.df_res['system_name'].unique()

        optimization_metrics_list = {}
        axs = {}
        for system_id in id_list:
            optimization_metrics, ax  = self.evaluate_one_individu(system_id)

            optimization_metrics_list[f'system_{system_id}'] = optimization_metrics
            axs[f'system_{system_id}'] = ax
        return optimization_metrics_list, axs
    
#%%

    
if __name__=='__main__':
    import logging
    from config import settings, create_psd_path, create_notch_path
    from psm.models.ad_systems import AD_GMM    
    from pathlib import Path
    from psm.utils.data.metadata import get_metadata_processed
    from psm.models.vanilla_classification import DenseSignalClassifierModule
    
    # Logging configuration
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Reading configuration and initializing paths.")
    settings_proc = 'SETTINGS1'
    settings_simu = 'SETTINGS1'
    root = Path(settings.data.path["processed"])

    database_path = create_psd_path(root,settings_proc,settings_simu)
    database_notch_path = create_notch_path(root,settings_proc,settings_simu)
    root_raw = Path(settings.data.path["raw"])
    model_paths = Path(settings.data.path['model']) / 'model'

    resonance_avg_path = root_raw/ settings_simu / 'resonance_frequency.csv'
    df_resonance_avg = pd.read_csv(resonance_avg_path)

    metadata = get_metadata_processed(settings_proc, settings_simu)
    freq_axis = metadata['freq']

    logging.info("Loading model from checkpoint.")
    name_model = 'best-epoch=28-val_loss=2.13.ckpt'
    model_path_1 = model_paths / name_model

    model = DenseSignalClassifierModule.load_from_checkpoint(model_path_1)

    ad_gmm = AD_GMM(num_classes=20, model=model.model)

    logging.info("Setting up data transformations and loading datasets.")
    from psm.models.prepare_data import CreateTransformer,PSDDataModule, PSDNotchDataset, PSDNotchDatasetOriginal

    # create the transformer
    transformer = CreateTransformer(database_path, freq=freq_axis, freq_min=0, freq_max=150)
    transform_psd = transformer.transform_psd
    transform_label = transformer.transform_label
    dm = PSDDataModule(database_path, transform_psd, transform_label, batch_size=32)
    dm.setup()
    train_dl = dm.train_dataloader()
    psd_notch = PSDNotchDataset(database_notch_path, transform=transform_psd, transform_label=transform_label)
    psd_original = PSDNotchDatasetOriginal(database_notch_path,transform=transform_psd, transform_label=transform_label)
    logging.info("Fitting AD system on training data.")
    ad_gmm.fit(train_dl)
    logging.info("Evaluating all individuals with Benchmark_VAS.")
    bench_vas = Benchmark_VAS(ad_gmm, psd_notch, psd_original, batch_size=500000, df_resonance_avg=df_resonance_avg)

    optimization_metrics_list, axs = bench_vas.evaluate_all_individus()
#%%
    logging.info("Evaluation results",)
    ax = axs['system_1']
    ax.get_figure().savefig('test.png')
    plt.show()