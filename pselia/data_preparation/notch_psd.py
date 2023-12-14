import sqlite3
import argparse
import logging
import numpy as np
from pselia.config_elia import settings, load_processed_data_path, load_processed_data_path_vas
from pselia.utils import load_freq_axis
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sys import float_info
EPS = float_info.epsilon


logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s', 
                    filename='logs/process_notch_dataset.log',
                      filemode='w')


# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--settings', type=str, default='SETTINGS3')
# grid search parameters for the notch filter
parser.add_argument('--f_oi_grid', type=str, default='1,30,0.4')
parser.add_argument('--amp_grid', type=str, default='-0.5,0.55,0.05')
parser.add_argument('--length', type=float, default=1)
args = parser.parse_args()
### Parse arguments
settings_name = args.settings
aff_f_grid = np.arange(*[float(x) for x in args.f_oi_grid.split(',')])
aff_amp_grid = np.arange(*[float(x) for x in args.amp_grid.split(',')])
aff_amp_grid = np.round(aff_amp_grid,2)
length = args.length
logging.info(f'f_oi_grid: {aff_f_grid}, amp_grid: {aff_amp_grid}')
### End parse arguments
# Load database paths
database_processed_path = load_processed_data_path(settings_name)
database_processed_path_vas = load_processed_data_path_vas(settings_name)
import matplotlib.pyplot as plt
class VirtualAnomaly:
    def __init__(self,freq_axis,length):
        self.freq_axis = freq_axis
        self.length = length
    
    def construct_nodge(self,f_oi,amp):
        hann_len = np.sum(np.abs(self.freq_axis-f_oi)<=self.length/2)
        hann = 1 - np.hanning(hann_len)*amp

        window = np.ones(self.freq_axis.shape)
        window[np.abs(self.freq_axis-f_oi)<=self.length/2] = hann


        return window
    
    def add_notch(self,psd,window):
        if psd.ndim == 1:
            psd = psd.reshape(1,-1)
        psd = np.clip(psd, EPS, np.inf)
        log_psd = np.log(psd)
        #  normalize the psd  
        # let's compute the min and max when frequency is between 0 and 50 Hz
        min_psd = np.min(log_psd)
        max_psd = np.max(log_psd)
        log_psd_n =  (log_psd  - min_psd) / (max_psd - min_psd) 
        log_psd_n_aff =log_psd_n +(1-window)
  #clip the psd to be between 0 and 1
        # then unnormalize the psd
        psd_unnorm_aff = (log_psd_n_aff) * (max_psd - min_psd) + min_psd 
        res = np.exp(psd_unnorm_aff)

        return res
    def affect_psd(self,psd,aff_f, aff_amp):
        if np.any(psd>1e20):
            raise ValueError('psd is too large')
        window = self.construct_nodge(aff_f,aff_amp)
        psd_res = self.add_notch(psd,window)
        # let's check for inf 
        if np.any(psd_res > 1e20):
            raise ValueError('affected psd is too large')
                    
        return psd_res
class HandleTable:
    def __init__(self,conn):
        self.conn = conn
        self.c = conn.cursor()

    def check_data(self,data):
        assert len(data[0])==6
        # no NaN
        assert not np.any(np.isnan(data))

    def create_table(self,table_name):
        table_name = table_name.upper()
        assert table_name in ['VAS_NOTCH','ORIGINAL_PSD']
        #let's delete the table if it exists
        self.c.execute(f"DROP TABLE IF EXISTS {table_name}")
        self.c.execute(f"""CREATE TABLE {table_name}(
            id INTEGER PRIMARY KEY,
            date_time DATETIME,
            PSD BLOB,
            direction TEXT,
            face TEXT,
            aff_f REAL,
            aff_amp REAL
        )""")
    def insert_data(self,data,table_name):
        table_name = table_name.upper()
        assert table_name in ['VAS_NOTCH','ORIGINAL_PSD']
        self.c.executemany(f"""INSERT INTO {table_name}  
                      (date_time, PSD, direction, face, aff_f, aff_amp)
                                                    VALUES (?,?,?,?,?,?)""",data)

freq_axis = load_freq_axis(database_processed_path)
vas = VirtualAnomaly(freq_axis,length=length)

def main():
    logging.info('Start processing the notch dataset')
    with sqlite3.connect(database_processed_path) as conn, \
        sqlite3.connect(database_processed_path_vas) as conn_vas:

        c = conn.cursor()
        c.execute("SELECT date_time, PSD, direction, face FROM processed_data WHERE stage='testing'")

        vas_table = HandleTable(conn_vas)
        logging.info('Create the table')
        vas_table.create_table('VAS_NOTCH')
        vas_table.create_table('ORIGINAL_PSD')
        logging.info('table created')
        data = c.fetchall()
        logging.info(f'Number of data points: {len(data)}')
        date_times, psds, directions, faces = zip(*[(
        date_time, np.frombuffer(psd, dtype=np.float64),direction, face)
                    for date_time, psd, direction, face in data])

        sensor_name = np.char.add(np.array(faces), np.array(directions))
        # let's split the data into train and test
        _, data_held = train_test_split(data, test_size=0.4, random_state=42, stratify=sensor_name)
        date_times , psds, directions, faces = zip(*[(
        date_time,np.frombuffer(psd, dtype=np.float64), direction,face) 
            for date_time, psd, direction, face in data_held])
        date_times = np.array(date_times)
        directions = np.array(directions)
        faces = np.array(faces)
        psds = np.array(psds)
        # let's insert the data into the database original psd
        data_to_insert =[(date_times[i],
                          psds[i].tobytes(),
                          directions[i],
                          faces[i],
                          0,
                          0) for i in range(len(date_times))]
        vas_table.insert_data(data_to_insert,'ORIGINAL_PSD')
        conn_vas.commit()
        progress_bar = tqdm(total=len(aff_f_grid)*len(aff_amp_grid), desc='Progress', position=0, leave=True)
        for f_oi in aff_f_grid:
            for amp in aff_amp_grid:
                psds_affected = vas.affect_psd(psds,aff_f = f_oi ,aff_amp = amp)
                data_to_insert =[(date_times[i],
                                  psds_affected[i].tobytes(),
                                  directions[i],
                                  faces[i],
                                  f_oi,
                                  amp) for i in range(len(date_times))]
                vas_table.insert_data(data_to_insert,'VAS_NOTCH')
                progress_bar.update(1)
                progress_bar.set_postfix({'f_oi':f_oi,'amp':amp})
                conn_vas.commit()
    


if __name__=='__main__':
    main()
    if False:
    

        import matplotlib.pyplot as plt
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
        c.execute("SELECT PSD FROM processed_data WHERE face='2' AND direction='Y' LIMIT 300")
        # process the data from buffer to numpy array
        psds = c.fetchall()
        psds = np.array([np.frombuffer(psd, dtype=np.float64) for psd, in psds[295:]])
        # let's load the freq axis
        freq = load_freq_axis(database_processed_path)
        vas = VirtualAnomaly(freq,length=length)
        # let's plot the psd
        fig,ax = plt.subplots(ncols=2)
        plot_psd(freq,psds.T,ax=ax[0],label='original')
        # let's affect the psd
        psds_affected = vas.affect_psd(psds,15,-0.5)
        plot_psd(freq,psds_affected.T,ax=ax[1],label='affected')

        fig.legend()
        plt.show()





