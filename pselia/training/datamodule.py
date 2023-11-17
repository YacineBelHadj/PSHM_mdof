import sqlite3
from typing import Union, Tuple, Callable, List
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
class CreateTransformer:
    """ this class intend to generate the transformer that you need to transform the psd
    and labels
    It also grab all psds and check the min and max for you
    processing consistent of :
    - cut the psd between freq_min and freq_max
    - take the log of the psd
    - normalize the psd between 0 and 1
    here is an example of how to use it:
    >>> transformer = PSDTransformer(database_path, freq, freq_min, freq_max)
    >>> transform_psd = transformer.transform_psd
    >>> label_transform = transformer.label_transform
    """
    def __init__(self, database_path:Union[str,Path], freq:np.ndarray, freq_min:float, freq_max:float, table_name:str='processed_data'):
        self.database_path = database_path
        self.freq = freq
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.min_psd, self.max_psd = self._calculate_min_max()
        assert self.min_psd < self.max_psd
        self.transform_psd = self._get_transform_psd()
        self.transform_label = self._get_transform_label()
        self.direction_map = {'X':0, 'Y':1, 'Z':2}
        self.face_map = {'2':0, '3':1, '4':2, '5':3}
        self.table_name = table_name    
        self.check_length_freq()
    
    def check_length_freq(self):
        # load 1 psd to check the length
        conn = sqlite3.connect(self.database_path)
        c = conn.cursor()
        c.execute("SELECT PSD FROM processed_data LIMIT 1")
        fetched_data = c.fetchone()
        psd = np.frombuffer(fetched_data[0],dtype=np.float64)
        assert len(psd) == len(self.freq)
      
    @property
    def new_freq(self):
        mask = (self.freq >= self.freq_min) & (self.freq <= self.freq_max)

        return self.freq[mask]

    @staticmethod
    def _cut_psd(psd:np.ndarray, freq:np.ndarray, freq_min:float, freq_max:float):
        mask = (freq >= freq_min) & (freq <= freq_max)
        return freq[mask], psd[mask]
    
    def _pre_transform(self, psd:np.ndarray):

        _, psd = self._cut_psd(psd, self.freq, self.freq_min, self.freq_max)
        psd = np.log(psd)

        return psd
    
    def _calculate_min_max(self):
  
        conn = sqlite3.connect(self.database_path)
        c = conn.cursor()
        c.execute("SELECT PSD FROM processed_data WHERE stage='training'")
        fetched_data = c.fetchall()
        psds = [self._pre_transform(np.frombuffer(psd[0])) for psd in fetched_data]

        return np.min(psds), np.max(psds)
        
    
    def _get_transform_psd(self):
        def transform_psd(psd):
            psd = self._pre_transform(psd)
            if np.any(np.isnan(psd)) or np.any(np.isinf(psd)):
                raise ValueError('there is a nan value in the psd')
            psd = (psd - self.min_psd) / (self.max_psd - self.min_psd)
            # check if there is a nan value

            return psd
        return transform_psd
    
    def _get_transform_label(self):
        def transform_label(face, direction ):
            direction = self.direction_map[direction]
            face = self.face_map[face]
            return face  ,direction
        return transform_label
    
    def dimension_psd(self):
        conn = sqlite3.connect(self.database_path)
        c = conn.cursor()
        c.execute("SELECT PSD FROM processed_data")
        data = np.frombuffer(c.fetchone()[0],dtype=np.float64)
        tr = self._get_transform_psd()
        data = tr(data)
        return len(data)


##############################################################################################################
import sqlite3
import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader

def build_query(stage: str, anomaly_description: List[str] = None) -> str:
    assert stage in ['anomaly', 'training', 'testing','all']

    if stage == 'all':
        return "SELECT id FROM processed_data"
    query = f"SELECT id FROM processed_data WHERE stage='{stage}'"
    if stage == 'anomaly' and anomaly_description is not None:
        query += f" AND anomaly_description IN ({','.join(['?']*len(anomaly_description))})"
    return query
    
class PSDELiaDataset(Dataset):
    """ a class that generate the dateset, it output a tuple (psd, direction,face)
    usage example:
    dataset = PSDELiaDataset(database_path,stage='training',
                              anomaly_description='start_measurement',)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    itere = iter(dataloader)
    """
    def __init__(self,database_path: Union[str, Path],
                 stage: str='all', anomaly_description: List[str]=None,
                 transform:Callable= None, label_transform:Callable=None,
                 preload:bool=False,return_mode='all'):
        assert return_mode in ['all','psd']
        self.return_mode = return_mode
        self.database_path = database_path
        self.stage = stage
        self.anomaly_description = anomaly_description
        self.transform = transform
        self.label_transform = label_transform
        self.preload = preload
        self.conn = sqlite3.connect(self.database_path)
        self.c = self.conn.cursor()
        self.query = build_query(stage=self.stage, anomaly_description=self.anomaly_description)
        self.c.execute(self.query)
        id_list = self.c.fetchall()
        self.keys = [d[0] for d in id_list]
        if self.preload:
            self.data = self._preload()

    def __len__(self):
        return len(self.keys)
    
    def _preload_data(self):
        keys_str = ','.join(['?']*len(self.keys))
        select_sql = f"SELECT PSD,direction,face FROM processed_data WHERE id IN ({keys_str})"
        self.c.execute(select_sql,self.keys)
        fetched_data = self.c.fetchall()
        return fetched_data
    
    def __getitem__(self, index:int):
        if self.preload:
            row = self.data[index]
        else:
            self.c.execute("SELECT PSD,direction,face FROM processed_data WHERE id=?",\
                           (self.keys[index],))
            row = self.c.fetchone()
        psd = np.frombuffer(row[0],dtype=np.float64)
        # to np.float32
        psd = psd.astype(np.float32)
        direction = row[1]
        face = row[2]
        if self.transform:
            psd = self.transform(psd)
        if self.label_transform:
            face, direction = self.label_transform(face, direction)

        if self.return_mode =='all':
            return psd, face, direction
        elif self.return_mode =='psd':
            return psd 


    
import pytorch_lightning as pl
import functools
from torch.utils.data.dataset import random_split

class PSDELiaDataModule(pl.LightningDataModule):
    """ a class that define dataloader for training and testing 
    usage example:
    dm = PSDELiaDataModule(database_path, batch_size=32, num_workers=4,
                            transform=transform_psd, label_transform=label_transform)
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    """
    def __init__(self, database_path: Union[str, Path], batch_size: int=64, num_workers: int=1,
                 transform:Callable= None, label_transform:Callable=None, val_split:float=0.2,
                 preload:bool=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = functools.partial(PSDELiaDataset, database_path=database_path,
                                            transform=transform, label_transform=label_transform,
                                            preload=preload)
        self.val_split = val_split
        
    def setup(self,stage=None):
        self.train_val_ds = self.dataset(stage='training')
        len_train_val_ds = len(self.train_val_ds)
        len_train = int(len_train_val_ds*(1-self.val_split))
        len_val = len_train_val_ds - len_train
        self.test_ds = self.dataset(stage='testing')
        self.train_ds, self.val_ds = random_split(self.train_val_ds,[len_train,len_val])
        self.ad_system_ds = self.dataset(stage='training',return_mode='psd')
            
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, 
                          num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, 
                          num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, 
                          num_workers=self.num_workers, shuffle=False)

    def ad_system_dataloader(self,batch_size:int=None):
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(self.ad_system_ds, batch_size=self.batch_size, 
                          num_workers=self.num_workers, shuffle=False)
    
if __name__=='__main__':
    # let's test the code
    from pselia.config_elia import settings, load_processed_data_path
    from pselia.utils import load_freq_axis
    from pathlib import Path
    import numpy as np

    database_path = load_processed_data_path('SETTINGS1')
    freq = load_freq_axis(database_path)
    freq_min , freq_max = settings.neuralnetwork.settings1.freq_range
    transformer = CreateTransformer(database_path, freq, freq_min=freq_min, freq_max=freq_max)
    transform_psd = transformer.transform_psd
    dim = transformer.dimension_psd()
    dm = PSDELiaDataModule(database_path, batch_size=32, num_workers=1,
                            transform=transform_psd, label_transform=transformer.transform_label)
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    for d in train_loader:
        mask_z_direction = np.array(d[1])=="Z"
        data = d[0][mask_z_direction].numpy()
        print(d[1])
        print(d[2])
        break

#### create the dataset for evaluation on the anomalies

class PSDELiaDataset_test(Dataset):
    def __init__(self,database_path:Union[str,Path], anomaly_description:List[str]=None,
                 stage:str='all', 
                 transform:Callable=None, label_transform:Callable=None,
                 preload:bool=False):
        super().__init__()
        self.database_path = database_path
        self.transform = transform
        self.label_transform = label_transform
        self.preload = preload
        self.c = sqlite3.connect(database_path).cursor()
        self.stage = stage
        self.anomaly_description = anomaly_description
        self.keys = self._get_keys()
        if self.preload:
            self.data = self._preload()
    
    def __len__(self):
        return len(self.keys)
    
    def _get_keys(self):
        query = build_query(stage=self.stage, anomaly_description=self.anomaly_description)
        self.c.execute(query)
        keys = self.c.fetchall()
        keys = [k[0] for k in keys]
        return keys
    
    def _preload(self):
        keys_str = ','.join(['?']*len(self.keys))
        select_sql = f"""SELECT psd, direction, face, date_time, stage, anomaly_description 
        FROM processed_data WHERE id IN ({keys_str})"""
        self.c.execute(select_sql,self.keys)
        fetched_data = self.c.fetchall()
        return fetched_data

    def __getitem__(self,index):
        if self.preload:
            row= self.data[index]
        else:
            self.c.execute("""SELECT psd, direction, face, date_time, stage, anomaly_description
                            FROM processed_data WHERE id = ?""",
                           (self.keys[index],))
            row = self.c.fetchone()
        psd = np.frombuffer(row[0],dtype=np.float64)
        # to np.float32
        psd = psd.astype(np.float32)
        direction = row[1]
        face = row[2]
        date_time = row[3]

        stage = row[4]
        anomaly_description = row[5]


        
        if self.transform:
            psd = self.transform(psd)
        if self.label_transform:
            face, direction = self.label_transform(face, direction)
        # let's check if there is a none value
        if date_time is None :
            date_time = 'None'


        return psd, face, direction, date_time, stage, anomaly_description
    

class PSDNotchDataset(Dataset):
    def __init__(self,database_path:Union[str,Path],transform=None,label_transform=None
                 ,original_psd:bool=False):
        super().__init__()
        self.database_path = database_path
        self.transform = transform
        self.label_transform = label_transform
        self.c = sqlite3.connect(database_path).cursor()
        self.table_name = 'ORIGINAL_PSD' if original_psd else 'VAS_NOTCH'
        query = f"SELECT id FROM {self.table_name}"
        self.c.execute(query)
        keys = self.c.fetchall()
        self.keys = [k[0] for k in keys]

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self,index):

        self.c.execute(f"SELECT PSD,direction,face, aff_f , aff_amp, date_time\
                        FROM {self.table_name} WHERE id=?",\
                        (self.keys[index],))
        row = self.c.fetchone()
        psd = np.frombuffer(row[0],dtype=np.float64)
        psd = psd.astype(np.float32)
        direction = row[1]
        face = row[2]
        f_affected = row[3]
        amplitude_notch = row[4]
        date_time = row[5]
        if self.transform:
            psd = self.transform(psd)
        if self.label_transform:
            face, direction = self.label_transform(face, direction)
        return psd, face, direction , f_affected, amplitude_notch, date_time
