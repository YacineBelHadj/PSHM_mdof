""" IN this file we process the data resulting from the simulation and saved in the data/raw/dataset.db file.
we compute the psds and save them in another database 
along each psd we save the resonance frequency, the anomaly level, 
the latent value and the amplitude of the excitation
"""
# first let's load the paths and the settings
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
raw_data_path = Path(settings.data.path['raw'])
process_data_path = Path(settings.data['path']['processed'])

database_raw_path = (raw_data_path/args.simulation/args.simulation.lower()).with_suffix('.db')
database_processed_path = (process_data_path/args.simulation/args.processing.lower()).with_suffix('.db')
#####
params_s = settings.simulation[args.simulation]
params_p = settings.processing[args.processing]
#####
sampling_frequency = 1/ params_s['simu_params']['dt']
filter_order = params_p['filter_params']['order']
lpf = params_p['filter_params']['upper']
nperseg = params_p['nperseg']
SNR = params_p['SNR']
TDD_length = params_s['simu_params']['t_end'] * sampling_frequency
loc_acc = params_p['acc_loc']
base_rms = params_s['resulting_mean_rms']

def create_database(c):
    # first delete the table if it exists
    c.execute("DROP TABLE IF EXISTS processed_data")
    c.execute("DROP TABLE IF EXISTS metadata")
    # create the table
    c.execute("""CREATE TABLE IF NOT EXISTS processed_data(
            id INTEGER PRIMARY KEY,
            simulation_name TEXT,
            system_name TEXT,
            PSD BLOB,
            resonance_frequency BLOB,
            anomaly_level REAL,
            latent_value REAL,
            RMS REAL,
            excitation_amplitude REAL,
            stage TEXT
            
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS metadata(
            fs REAL,
            nperseg REAL,
            filter_order REAL,
            lpf REAL,
            freq BLOB,
            SNR REAL
              )""")

def main():
    # compute min and max of psd 
    database_processed_path.parent.mkdir(parents=True, exist_ok=True)
    conn_raw = sqlite3.connect(database_raw_path)
    conn_processed = sqlite3.connect(database_processed_path)

    c_raw = conn_raw.cursor()
    c_processed = conn_processed.cursor()

    create_database(c_processed)
    c_raw.execute("SELECT COUNT(*) FROM simulation")
    n_simu = c_raw.fetchone()[0]
    logging.info(f"Number of simulation to process: {n_simu}")

    chunksize = 1000
    for i in tqdm(range(0, n_simu, chunksize),desc='Processing data'):
        c_raw.execute(f"""SELECT id, name, TDD_output, resonance_frequency, 
                      anomaly_level,stage, latent, amplitude FROM simulation
                       LIMIT {chunksize} OFFSET {i}""")
        
        rows = c_raw.fetchall()
        for row in rows:
            sim_id = row[0]
            system_name = row[1]
            TDD_output = np.frombuffer(row[2], dtype=np.float64)
            res_freq = np.frombuffer(row[3], dtype=np.float64)
            anomaly_level = row[4]
            stage = row[5]
            latent = row[6]
            amplitude = row[7]
            TDD_output = TDD_output.reshape(int(TDD_length), -1)
            acc7 = TDD_output[:, loc_acc]
            acc7 = proc.add_noise(acc7, SNR)
            acc7_f, rms = proc.preprocess_vibration_data(acc7, filter_order, lpf, sampling_frequency)
            f, Sxx = proc.apply_welch(acc7_f, sampling_frequency, nperseg=nperseg)
            
            c_processed.execute("""INSERT INTO processed_data(
                                simulation_name,
                                system_name,
                                PSD,
                                resonance_frequency,
                                anomaly_level,
                                stage,
                                latent_value,
                                RMS,
                                excitation_amplitude)
                                VALUES(?,?,?,?,?,?,?,?,?)""",
                                 (sim_id,
                                  system_name,  
                                  Sxx.tobytes(),
                                  res_freq.tobytes(),
                                  anomaly_level,
                                  stage,
                                  latent,
                                  rms,
                                  amplitude))
        conn_processed.commit()

    c_processed.execute("""INSERT INTO metadata(
                        fs,nperseg, filter_order,
                        lpf, freq, SNR)
                        VALUES(?,?,?,?,?,?)""",
                        (sampling_frequency,
                        nperseg,
                        filter_order,
                        lpf,
                        f.tobytes(),
                        SNR))
    conn_processed.commit()

if __name__ == '__main__':
    main()

