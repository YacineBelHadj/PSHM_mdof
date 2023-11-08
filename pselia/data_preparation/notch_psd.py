import sqlite3
import argparse
import logging
import numpy as np
from pselia.config_elia import settings, load_processed_data_path, load_processed_data_path_vas
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s', 
                    filename='logs/process_notch_dataset.log',
                      filemode='w')

# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--settings', type=str, default='SETTINGS1')
args = parser.parse_args()
settings_name = args.settings

# Load database paths
database_processed_path = load_processed_data_path(settings_name)
database_processed_path_vas = load_processed_data_path_vas(settings_name)
def load_freq_axis(database_processed_path):
    conn = sqlite3.connect(database_processed_path)
    c = conn.cursor()
    c.execute("SELECT freq FROM metadata")
    freq = c.fetchone()[0]
    freq = np.frombuffer(freq, dtype=np.float64)
    return freq

def construct_nodge(freq,f_oi,length,amp):
    hann_len = np.sum(np.abs(freq-f_oi)<=length/2)
    hann = 1-np.hanning(hann_len)*amp
    window = np.ones(freq.shape)
    window[np.abs(freq-f_oi)<=length/2] = hann
    return window

def multiply_signals_log(psd,window):
    log_psd = np.log(psd)
    #  normalize the psd
    min_psd = np.min(log_psd)
    max_psd = np.max(log_psd)
    log_psd_n = (log_psd - min_psd) / (max_psd - min_psd)

    log_psd_n_aff =log_psd_n+ (1- window)
    # then unnormalize the psd
    psd_unnorm_aff = log_psd_n_aff * (max_psd - min_psd) + min_psd
    res = np.exp(psd_unnorm_aff)
    return res
def add_notch(psd,window):
    if psd.ndim==1:
        psd = psd.reshape(1,-1)
    psd_log = np.log(psd)
    min_psd = np.min(psd_log,axis=-1,keepdims=True)
    max_psd = np.max(psd_log,axis=-1,keepdims=True)
    psd_log_n = (psd - min_psd) / (max_psd - min_psd)
    psd_log_n_aff = psd_log_n + (1-window) 
    psd_unnorm_aff = psd_log_n_aff * (max_psd - min_psd) + min_psd
    res = np.exp(psd_unnorm_aff)
    return res

def affect_psd(psd,freq,amp,f_oi,length):
    window = construct_nodge(freq=freq,f_oi=f_oi,length=length,amp=amp)
    psd_res = multiply_signals_log(psd,window)
    return psd_res

def create_table_VAS_notch(c):
    c.execute("""CREATE TABLE VAS_NOTCH(
        id INTEGER PRIMARY KEY,
        date_time DATETIME,
        PSD BLOB,
        direction TEXT,
        face TEXT,
        f_affected REAL,
        amplitude_notch REAL
    )""")
    c.execute("""CREATE TABLE ORIGINAL_PSD(
        id INTEGER PRIMARY KEY,
        date_time DATETIME,
        PSD BLOB,
        direction TEXT,
        face TEXT,
        f_affected REAL,
        amplitude_notch REAL
    )""")

def main():
    with sqlite3.connect(database_processed_path) as conn, \
        sqlite3.connect(database_processed_path_vas) as conn_vas:

        c = conn.cursor()
        c_vas = conn_vas.cursor()

        create_table_VAS_notch(c_vas)

        freq = load_freq_axis(database_processed_path)
        # query the data where the stage is testing and the direction is Z
        c.execute("SELECT date_time, PSD, direction, face FROM processed_data WHERE stage='testing' AND direction='Z'")
        data = c.fetchall()

        date_times, psds, directions, faces = zip(*[(
            date_time, 
            np.frombuffer(psd, dtype=np.float64), 
            direction, 
            face
        ) for date_time, psd, direction, face in data])
        date_times = np.array(date_times)
        psds = np.array(psds)
        directions = np.array(directions)
        faces = np.array(faces)
        f_oi_grid = np.arange(1,50,1)
        amp_grid = np.arange(-0.2,0.25,0.05)
        amp_grid = np.round(amp_grid,2)
        print(f'f_oi_grid: {f_oi_grid}, amp_grid: {amp_grid}')
        progress_bar = tqdm(total=len(f_oi_grid)*len(amp_grid), desc='Progress', position=0, leave=True)
        for f_oi in f_oi_grid:
            for amp in amp_grid:
                psds_affected = affect_psd(psds,freq,amp,f_oi,length=1)
                data_to_insert =[(date_times[i],
                                  sqlite3.Binary(psds_affected[i].tobytes()),
                                  directions[i],
                                  faces[i],
                                  f_oi,
                                  amp) for i in range(len(date_times))]
                c_vas.executemany("""INSERT INTO VAS_NOTCH (date_time, PSD, direction, face, f_affected, amplitude_notch)
                                                VALUES (?,?,?,?,?,?)""",data_to_insert)
                progress_bar.update(1)
                progress_bar.set_postfix({'f_oi':f_oi,'amp':amp})
        conn_vas.commit()
    


if __name__=='__main__':
    main()







    if False:



        def plot_psd(freq,psd,ylog=True,ax:plt.Axes=None,label=None):
            if ax is None:
                fig,ax = plt.subplots()
            
            ax.plot(freq,psd,label=label)
            if ylog:
                ax.set_yscale('log')
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('PSD [g**2/Hz]')
            ax.grid(True, which='both')
            ax.set_title('PSD')
            if ax is None:
                plt.show()
                plt.close()
            return ax
        # let's load one psd from the database
        conn = sqlite3.connect(database_processed_path)
        c = conn.cursor()
        # let's load 10 psd where the face is 1 and the direction is X
        c.execute("SELECT PSD FROM processed_data WHERE face='2' AND direction='Z' LIMIT 1")
        # process the data from buffer to numpy array
        psds = c.fetchall()
        psds = np.array([np.frombuffer(psd, dtype=np.float64) for psd, in psds])
        # let's load the freq axis
        freq = load_freq_axis(database_processed_path)
        # let's plot the psd
        fig,ax = plt.subplots(ncols=2)
        plot_psd(freq,psds.T,ax=ax[0],label='original')
        # let's affect the psd
        nodge_filter = construct_nodge(freq,f_oi=7, length=1,amp = 0.1)
        ax[1].plot(freq,1-nodge_filter)

        psds_affected  = multiply_signals_log(psds,nodge_filter)
        plot_psd(freq,psds_affected.T,ax=ax[0],label='affected')

        fig.legend()
        plt.show()
        




