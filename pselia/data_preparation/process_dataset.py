import sqlite3
import numpy as np
import logging
from pselia.config_elia import settings, load_processed_data_path, load_measurement_bound
from processing import preprocess_vibration_data, apply_welch
from utils_labelling import get_event_state_stage
import load_data
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler("logs/processing_elia.log")])

parser = argparse.ArgumentParser(description='Process elia data')
parser.add_argument('--settings',type=str,\
                    help='settings_param name in configuration/settings_elia.toml'\
                        ,default='SETTINGS2')
args = parser.parse_args()

class Config():
    """ Configuration class to store all the configuration parameters"""
    def __init__(self, settings, args):
        self.raw_data_path = Path(settings.dataelia.path['raw'])
        self.process_data_path = Path(settings.dataelia['path']['processed'])
        self.database_processed_path = load_processed_data_path(args.settings)
        self.fs = settings.dataelia.sensor['fs']
        self.params_p = settings.processing[args.settings]
        self.filter_order = self.params_p['filter_params']['order']
        self.lpf = self.params_p['filter_params']['lpf']
        self.nperseg = self.params_p['nperseg']
        self.duration = timedelta(minutes=self.params_p['duration'])
        self.step = timedelta(minutes=self.params_p['step'])
        self.start_time, self.end_time= load_measurement_bound()
        self.freq = None

def create_database(c):
    # ask the user if he wants to delete the database
    # if yes, delete the database and create a new one
    # if no, abort the program
    # check the database exists
    c.execute("DROP TABLE IF EXISTS processed_data")
    c.execute("DROP TABLE IF EXISTS metadata")
    c.execute("""CREATE TABLE IF NOT EXISTS processed_data(
            id INTEGER PRIMARY KEY,
            date_time DATETIME,
            PSD BLOB,
            RMS DOUBLE,
            direction TEXT,
            face TEXT,
            stage TEXT,
            state TEXT, 
            anomaly_description TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS metadata(
            freq BLOB,
            nperseg REAL,
            filter_order REAL,
            lpf REAL
    )""")

def extract_sensor_info(sensor_name:str):
    """Extracts direction and face of sensor from sensor name"""
    sensor_name_temp = sensor_name.split('_')
    face = sensor_name_temp[0][-1]
    direction = sensor_name_temp[1]
    return face, direction
def extract_sensor_list_info(sensor_name:list):
    """Extracts direction and face of sensor from sensor name"""
    face, direction = zip(*[extract_sensor_info(s) for s in sensor_name])
    face = np.array(face)
    direction = np.array(direction)
    return face, direction

def remove_sensor_1(sensor_name, signals):
    """Removes sensor 1 from the data"""
    mask_sensor_1 = np.array([s.split('_')[0][-1] == '1' for s in sensor_name])
    signals = signals[~mask_sensor_1]
    sensor_name = sensor_name[~mask_sensor_1]
    return sensor_name, signals

def process_data(data,config):
    """Processes vibration data"""
    sensor_name, signals = zip(*data.items())
    sensor_name = np.array(sensor_name)
    signals = np.array(signals)

    sensor_name, signals = remove_sensor_1(sensor_name, signals)
    face, direction = extract_sensor_list_info(sensor_name)
    # preprocess the data
    singals_preprocessed ,rms = preprocess_vibration_data(signals, 
                                                          filter_order=config_.filter_order,
                                                          lpf=config_.lpf,
                                                          sampling_frequency=config_.fs)
    freq, psd = apply_welch(singals_preprocessed,
                            config_.fs,
                            nperseg=config_.nperseg)
    freq_mask =  (freq<=config_.lpf+20)

    res_dict = {'freq':freq[freq_mask],
                'psd':psd[:,freq_mask],
                'rms':rms,
                'face':face,
                'direction':direction}
    return res_dict

def process_dt(dt,length_data):
    event, state, stage = get_event_state_stage(dt)
    # replicate the event name, event state and stage name for each sensor
    event = [event] * length_data
    state = [state] * length_data
    stage = [stage] * length_data
    res_dict = {'event':event,
                'state':state,
                'stage':stage}
    return res_dict

def insert_proccessed_data(c_processed, data):
    insert_sql  = """
    INSERT INTO processed_data
    (date_time, PSD,RMS,direction,face,stage,state,anomaly_description)
    VALUES (?,?,?,?,?,?,?,?)
    """
    c_processed.executemany(insert_sql,data)

def insert_metadata(c_processed, metadata):
    insert_sql = """
    INSERT INTO metadata
    (freq, nperseg, filter_order, lpf)
    VALUES (?,?,?,?)
    """
    c_processed.execute(insert_sql,metadata)
    

def process_iteration(loader,dt,c_processed,progress_bar,config):
    dt_init = dt
    data = loader.get_data(start=dt, end=dt+config.duration)
    if data is None:
        logging.info(f'No data at {dt}')
        dt += config.step
        progress_bar.update(1)
        progress_bar.set_postfix({'dt':dt})
        return dt 
    # process the data

    res_dict = process_data(data,config=config)
    # process the datetime
    res_dict_dt = process_dt(dt,length_data=len(res_dict['face']))
    # merge the two dictionaries
    res_dict.update(res_dict_dt)
    # convert the data to a list of tuples
    data_to_insert = [(dt, 
                        sqlite3.Binary(res_dict['psd'][i].tobytes()), 
                        res_dict['rms'][i], 
                        res_dict['direction'][i], 
                        res_dict['face'][i], 
                        res_dict['stage'][i], 
                        res_dict['state'][i], 
                        res_dict['event'][i]) for i in range(len(res_dict['face']))]
    # insert the data into the database
    insert_proccessed_data(c_processed, data_to_insert)
    progress_bar.update(1)
    progress_bar.set_postfix({'dt':dt,
                              'stage':res_dict['stage'][0]})
    dt += config.step
    config.freq = res_dict['freq']
    # assert that dt increased by step
    assert dt == dt_init + config.step
    return dt


def process_loop(config):
    conn_processed = sqlite3.connect(config.database_processed_path)
    c_processed = conn_processed.cursor()
    create_database(c_processed)
    sensor = load_data.Sensor(name='ACC', location='MO04', data_type='TDD', format='.tdms')
    loader = load_data.DataLoader(sensor=sensor, data_root=config.raw_data_path)
    total_steps = int((config.end_time - config.start_time)/config.step)
    progress_bar = tqdm(total=total_steps, desc='Processing data')
    dt = config.start_time
    while dt < config.end_time:   
        dt = process_iteration(loader, dt, c_processed, progress_bar, config)
                    

    conn_processed.commit()
    # insert metadata into the database
    freq_blob = sqlite3.Binary(config.freq.tobytes())
    metadata = (freq_blob, config.nperseg, config.filter_order, config.lpf)
    insert_metadata(c_processed, metadata)
    conn_processed.commit()


if __name__=='__main__':
    config_ = Config(settings,args)
    process_loop(config=config_)
