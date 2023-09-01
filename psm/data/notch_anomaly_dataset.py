from psm.data import processing as proc
import sqlite3  
import pandas as pd
import numpy as np
from scipy.signal import welch
from tqdm import tqdm
from config import settings
from pathlib import Path
import logging
import argparse
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename='logs/process_dataset.log',
                    filemode='w')
argparse = argparse.ArgumentParser()
argparse.add_argument('--simulation', type=str, default='SETTINGS1')
argparse.add_argument('--processing', type=str, default='SETTINGS1')
args = argparse.parse_args()

#####
process_data_path = Path(settings.data['path']['processed'])

database_processed_path = (process_data_path/args.simulation/args.processing.lower()).with_suffix('.db')
database_vas = (process_data_path/args.simulation/(args.processing.lower()+'_vas')).with_suffix('.db')
#####
params_s = settings.simulation[args.simulation]
params_p = settings.processing[args.processing]
#####
sampling_frequency = 1/ params_s['simu_params']['dt']
filter_order = params_p['filter_params']['order']
lpf = params_p['filter_params']['upper']
nperseg = params_p['nperseg']
# load metadata from the processed database
def load_freq_axis(database_processed_path):
    conn = sqlite3.connect(database_processed_path)
    c = conn.cursor()
    c.execute("SELECT freq FROM metadata")
    freq = c.fetchone()[0]
    freq = np.frombuffer(freq, dtype=np.float64)
    return freq

def construct_nodge(freq,f_oi,length,amp):
    freq_index = len(freq)/freq[-1]
    length = np.round(length*freq_index,0).astype('int')
    window  = np.ones(freq.shape)
    hann_len = np.sum(np.abs(freq-f_oi)<length)
    hann = 1-np.hanning(hann_len)*amp
    window[np.abs(freq-f_oi)<length] = hann
    return window
def multiply_signals_log(psd,window):


    log_psd = np.log(psd)
    #  normalize the psd
    min_psd = np.min(log_psd)
    max_psd = np.max(log_psd)
    log_psd = (log_psd - min_psd) / (max_psd - min_psd)
    psd_aff =log_psd+(1-window)
    # then unnormalize the psd
    psd_aff = psd_aff * (max_psd - min_psd) + min_psd
    res = np.exp(psd_aff)
    return res
def affect_psd(psd,freq,amp,f_oi,length):
    window = construct_nodge(freq=freq,f_oi=f_oi,length=length,amp=amp)
    psd_res = multiply_signals_log(psd,window)
    return psd_res

def create_table_VAS_notch(c):
    c.execute("""CREATE TABLE IF NOT EXISTS VAS_notch(
            id INTEGER PRIMARY KEY,
            simulation_name TEXT,
            system_name TEXT,
            f_affected REAL,
            amplitude_notch REAL,
            PSD_notch BLOB
            ) """)
    c.execute("""CREATE TABLE ORIGINAL_PSD(
            id INTEGER PRIMARY KEY,
            simulation_name TEXT,
            system_name TEXT,
            PSD BLOB
    )""")


def main():
    
    conn_vas = sqlite3.connect(database_vas)
    c_vas = conn_vas.cursor()
    create_table_VAS_notch(c_vas)
    conn_psd = sqlite3.connect(database_processed_path)
    c_psd = conn_psd.cursor()
    freq = load_freq_axis(database_processed_path)
    f_affected_grid = np.arange(1,140,0.5)
    amp_grid = np.arange(-0.35,0.4,0.05)
    amp_grid = np.round(amp_grid,2)
    print(f'f_affected_grid: {f_affected_grid}, amp_grid: {amp_grid}')

    # fetch all data before the loops
    c_psd.execute("""SELECT simulation_name,PSD,system_name FROM processed_data
                  WHERE stage= 'test' 
                  """)
    data = c_psd.fetchall()
    for row in tqdm(data):


        simu_name = row[0]
        psd = np.frombuffer(row[1], dtype=np.float64)
        system_name = row[2]
        # save the original psd
        conn_vas.execute("""INSERT INTO ORIGINAL_PSD (simulation_name, system_name, PSD)
                                                VALUES (?,?,?)""",
        (simu_name, system_name, row[1]))

        for f_affected in f_affected_grid:
            for amp in amp_grid:
                psd_res_notched = affect_psd(psd=psd, freq=freq, amp=amp, f_oi=f_affected, length=1)
                # convert psd_res to bytes to store as a blob
                psd_res_notched = psd_res_notched.tobytes()
                f_affected = float(f_affected)
                conn_vas.execute("""INSERT INTO VAS_notch (simulation_name, system_name, f_affected, amplitude_notch, PSD_notch)
                                                VALUES (?,?,?,?,?)""",
                (simu_name, system_name,f_affected, amp, psd_res_notched))
                    
        conn_vas.commit()

if __name__=='__main__':
    main()
