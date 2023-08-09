import numpy as np
import sqlite3
from typing import Union, Tuple
from pathlib import Path


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
    def __init__(self, database_path: Union[str, Path], freq, freq_min, freq_max):
        self.database_path = database_path
        self.freq = freq
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.min_psd, self.max_psd = self._calculate_min_max()
        self.transform_psd = self._get_transform_psd()
        self.transform_label = self._get_transform_label()
    @property
    def new_freq(self):
        mask = (self.freq >= self.freq_min) & (self.freq <= self.freq_max)
        return self.freq[mask]
        
    @staticmethod
    def _cut_psd(freq, psd, freq_min, freq_max):
        mask = (freq >= freq_min) & (freq <= freq_max)
        return freq[mask], psd[mask]
    def _pre_transform(self, psd):
        f , psd = self._cut_psd(self.freq, psd, self.freq_min, self.freq_max)
        psd = np.log(psd)
        return psd
    def _calculate_min_max(self):
        def pre_transform(psd):
            _, psd = self._cut_psd(self.freq, psd, self.freq_min, self.freq_max)
            psd = np.log(psd)
            return psd

        conn = sqlite3.connect(self.database_path)
        c = conn.cursor()
        c.execute("SELECT PSD FROM processed_data WHERE anomaly_level=0")
        fetched_data = c.fetchall()
        psd = [np.frombuffer(d[0], dtype=np.float64) for d in fetched_data]
        psd = [pre_transform(p) for p in psd]
        return np.min(psd), np.max(psd)

    def _get_transform_psd(self):
        def transform_psd(psd):
            res = self._pre_transform(psd)
            res = (res - self.min_psd) / (self.max_psd - self.min_psd)
            return res
        return transform_psd
    

    def _get_transform_label(self):
        def transform_label(x):
            return int(x.split('_')[-1])
        return transform_label



    def dimension_psd(self):
        conn = sqlite3.connect(self.database_path)
        c = conn.cursor()
        c.execute("SELECT PSD FROM processed_data")
        data = np.frombuffer(c.fetchone()[0],dtype=np.float64)
        tr = self._get_transform_psd()
        data = tr(data)
        return len(data)

######################################
import sqlite3
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Union, List
def build_query(anomaly_level, system_name,stage):
    """ a utile function that prepare the query for you based on the anomaly_level and system_name
    """
    params = []
    clauses = []
    if anomaly_level is not None and anomaly_level != '*':
        if isinstance(anomaly_level, list):
            clause = " OR ".join("anomaly_level=?" for _ in anomaly_level)
            params.extend(anomaly_level)
        else:
            clause = "anomaly_level=?"
            params.append(anomaly_level)
        clauses.append(f"({clause})")
    if system_name is not None:
        clauses.append("system_name=?")
        params.append(system_name)
    if stage is not None:
        clauses.append("stage=?")
        params.append(stage)
    query = " AND ".join(clauses) if clauses else "1"  # If no conditions, use "1" to get all data
    return query, tuple(params)



class PSDDataset(Dataset):
    """ a class that generate the dataset, it output the psd and the label
    a possible modification is to add the key as output as well
    usage example:
    dataset = PSDDataset(database_path=database_path, anomaly_level=0,
                     transform_label=transform_label,
                     transform=transform_psd,
                     stage='train/val')
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    for batch in dataloader:
        psd, system_name = batch
        print(system_name)
        plt.plot(psd.T)
        plt.show()
        break   
    """
    def __init__(self, database_path: Union[str, Path], system_name: str = None,
                 anomaly_level: Union[float, List[float], str] = None,
                 transform=None, transform_label=None, preload: bool = False,
                 stage=None):
        
        self.database_path = database_path
        self.system_name = system_name
        self.transform = transform
        self.transform_label = transform_label
        self.preload = preload
        self.stage = stage
        self.conn = sqlite3.connect(self.database_path)
        self.c = self.conn.cursor()

        self.query, self.params = build_query(anomaly_level, system_name,stage)
        print (self.query, self.params)
        self.c.execute(f"SELECT id,stage FROM processed_data WHERE {self.query}", self.params)
        fetched_data = self.c.fetchall()
        self.keys = [d[0] for d in fetched_data]

        if self.preload:
            self.data = self._preload_data()

    def __len__(self):
        return len(self.keys)

    def _preload_data(self):
        # Convert list of keys to a format suitable for SQL IN keyword
        keys_str = ', '.join('?' for _ in self.keys)

        # Execute the query using the keys_str and self.keys
        self.c.execute(f"SELECT PSD, system_name, anomaly_level FROM processed_data WHERE id IN ({keys_str})", self.keys)
        data = self.c.fetchall()
        return data


    def __getitem__(self, index: int):
        if self.preload:
            row = self.data[index]
        else:
            self.c.execute(f"SELECT PSD, system_name,anomaly_level FROM processed_data WHERE id=?", (self.keys[index],))
            row = self.c.fetchone()
        psd = np.frombuffer(row[0], dtype=np.float64)
        system_name = row[1]
        anomaly_level = row[2]
        if self.transform:
            psd = self.transform(psd)
        if self.transform_label:
            system_name = self.transform_label(system_name)
        psd = torch.from_numpy(psd).float()
        
        if self.stage == 'anomaly':
            return psd, system_name, anomaly_level
        else :
            return psd, system_name
    

######################

import pytorch_lightning as pl  
import functools
from torch.utils.data.dataset import random_split
 
class PSDDataModule(pl.LightningDataModule):
    """ a class that define dataloader for training and testing
    usage example:
        # Instantiate the PSDDataModule
        dm = PSDDataModule(database_path=database_path,transform=transform_psd,
                        transform_label=transform_label, batch_size=64)

        # Call the setup method
        dm.setup()

        # Create dataloaders
        train_dl = dm.train_dataloader()
        val_dl = dm.val_dataloader()
        test_dl = dm.test_dataloader()

        # Print some details
        print(f"Number of training samples: {len(dm.train_dataset)}")
        print(f"Number of validation samples: {len(dm.val_dataset)}")
        print(f"Number of test samples: {len(dm.test_dataset)}")
 
        # Get a batch of data
        for batch in train_dl:
            data, label = batch
            print(f"Shape of data from train_dl: {data.shape}")
            print(f"Shape of label from train_dl: {label.shape}")
            break
    """
    def __init__(self, database_path, transform=None, transform_label=None, batch_size:int = 64
                 ,num_workers:int = 1):
        super().__init__()
        self.batch_size = batch_size
        self.dataset = functools.partial(PSDDataset, database_path=database_path,
                                         transform=transform, transform_label=transform_label)
        self.num_workers = num_workers

    def setup(self, stage=None): 
        self.full_train_dataset = self.dataset(anomaly_level=0, preload=True, stage='train')
        generator1 = torch.Generator().manual_seed(42)

        self.train_dataset, self.val_dataset = random_split(self.full_train_dataset,[800*20,200*20],
                                                            generator=generator1)
        self.test_dataset = self.dataset(anomaly_level=0, preload=True, stage='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
####

def build_query_system(system_name: str) -> Tuple[str, Tuple]:
    if system_name == '*':
        return "system_name LIKE ?", ('%',)
    else:
        return "system_name=?", (system_name,)

class PSDNotchDataset(Dataset):
    def __init__(self, database_path: Union[str, Path], system_name: str = '*', 
                 transform=None, transform_label=None, preload:bool=False):
        self.database_path = database_path
        self.system_name = system_name
        self.transform = transform
        self.transform_label = transform_label
        self.preload = preload
        self.conn = sqlite3.connect(self.database_path)
        self.c = self.conn.cursor()
        
        query, params = build_query_system(self.system_name)
        print (query, params)
        self.c.execute(f"SELECT id FROM VAS_notch WHERE {query}", params)
        self.keys = [row[0] for row in self.c.fetchall()]

        if self.preload:
            self.data = self._preload_data()

    def __len__(self):
        return len(self.keys)

    def _preload_data(self):
        # Convert list of keys to a format suitable for SQL IN keyword
        keys_str = ', '.join('?' for _ in self.keys)

        # Execute the query using the keys_str and self.keys
        self.c.execute(f"SELECT PSD_notch, system_name, amplitude_notch, f_affected FROM VAS_notch WHERE id IN ({keys_str})", self.keys)
        data = self.c.fetchall()
        return data

    def __getitem__(self,index:int):
        if self.preload:
            row = self.data[index]
        else:
            self.c.execute(f"SELECT PSD_notch, system_name, amplitude_notch, f_affected FROM VAS_notch WHERE id=?", (self.keys[index],))
            row = self.c.fetchone()

        psd = np.frombuffer(row[0], dtype=np.float64)
        system_name = row[1]
        amplitude_notch = row[2]
        f_affected = row[3]

        if self.transform:
            psd = self.transform(psd)

        if self.transform_label:
            system_name = self.transform_label(system_name)

        psd = torch.from_numpy(psd).float()

        return psd, system_name, amplitude_notch, f_affected
    
class PSDNotchDatasetOriginal(Dataset):
    def __init__(self,database_path : Union[str, Path], system_name : str = '*',
                    transform=None, transform_label=None, preload:bool=False):
                 
        self.database_path = database_path
        self.system_name = system_name
        self.transform = transform
        self.transform_label = transform_label
        self.preload = preload
        self.conn = sqlite3.connect(self.database_path)
        query, params = build_query_system(self.system_name)
        print (query, params)
        self.c = self.conn.cursor()
        self.c.execute(f"SELECT id FROM ORIGINAL_PSD WHERE {query}", params)
        self.keys = [row[0] for row in self.c.fetchall()]

        if self.preload:
            self.data = self._preload_data()
        
    def __len__(self):
        return len(self.keys)

    def _preload_data(self):
        # Convert list of keys to a format suitable for SQL IN keyword
        keys_str = ', '.join('?' for _ in self.keys)

        # Execute the query using the keys_str and self.keys
        self.c.execute(f"SELECT PSD, system_name FROM ORIGINAL_PSD WHERE id IN ({keys_str})", self.keys)
        data = self.c.fetchall()
        return data
    
    def __getitem__(self,index:int):
        if self.preload:
            row = self.data[index]
        else:
            self.c.execute(f"SELECT PSD, system_name FROM ORIGINAL_PSD WHERE id=?", (self.keys[index],))
            row = self.c.fetchone()

        psd = np.frombuffer(row[0], dtype=np.float64)
        system_name = row[1]

        if self.transform:
            psd = self.transform(psd)

        if self.transform_label:
            system_name = self.transform_label(system_name)

        psd = torch.from_numpy(psd).float()

        return psd, system_name , 0 , 0
        