""" IN this file we process the data resulting from the simulation and saved in the data/raw/dataset.db file.
we compute the psds and save them in another database 
along each psd we save the resonance frequency, the anomaly level, 
the latent value and the amplitude of the excitation
"""
# first let's load the paths and the settings
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
                    filename='log/process_dataset.log',
                    filemode='w')
argparse = argparse.ArgumentParser()
argparse.add_argument('--settings', type=str, default='SETTINGS1')
args = argparse.parse_args()

settings_simu = 'SETTINGS1'
settings_processing = 'SETTINGS1'
raw_data_path = Path(settings.data.path['raw'])
process_data_path = Path(settings.data['path']['processed'])

settings_config = settings.simulation[settings_simu]
database_raw_path = (raw_data_path/settings_simu/settings_simu.lower()).with_suffix('.db')
database_processed_path = (process_data_path/settings_processing/settings_processing.lower()).with_suffix('.db')

# now let's load the data the raw data from the database
