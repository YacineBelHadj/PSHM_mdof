import sqlite3
from typing import Union, Tuple, Callable
from pathlib import Path
import numpy as np

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
    def __init__(self, database_path:Union[str,Path], freq:np.ndarray, freq_min:float, freq_max:float):
        self.database_path = database_path
        self.freq = freq
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.min_psd = None
        self.max_psd = None
        self.transform_psd = None
        self.label_transform = None
        self._get_min_max()
        self._create_transformer()
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
        def pre_transform(psd):
            psd = self._cut_psd(psd, self.freq, self.freq_min, self.freq_max)
            psd = np.log(psd)
            return psd
        conn = sqlite3.connect(self.database_path)
        c = conn.cursor()
        c.execute("SELECT PSD FROM processed_data WHERE stage='training'")