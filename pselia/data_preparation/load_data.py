from utils import readTDMS, append_dict , datetime_to_path

from dataclasses import dataclass
from typing import Union
from pathlib import Path
from datetime import datetime,  timedelta
import pandas as pd
import numpy as np
from config import settings

@dataclass(frozen=True)
class Sensor:
    """Sensor class to store sensor information
    name : str
        Name of the sensor
    location : str
        location in which the sensor is installed 
    data_type : str
        type of data collected by the sensor
    format : str
        format of the data collected by the sensor
    """

    name: str ='ACC'
    location: str = 'MO04'
    data_type : str = 'TDD'
    format : str = '.tdms'

@dataclass
class DataLoader:
    """ Data loader class to load data from a file"""
    sensor: Sensor = None
    data_root = None
    path = None
    time_step = timedelta(minutes=1)
    data_root : Union[Path,str] = None

    def __post_init__(self):
        """ Load config file and set data root and path"""
        if not self.data_root.exists():
            print(f'Invalid data root: {self.data_root}')
        self.path = self.data_root / self.sensor.location /self.sensor.data_type

    def _load_single(self, dt):
        """Load data from a single file
        Parameters
        ----------
        dt : datetime
            datetime of the file to be read
        Returns
        -------
        data : dict
            data from the file
        """
        path = datetime_to_path(self.path, dt,self.sensor.format)
        if not path.exists():
            print(f'No data at {path}')
            return None
        data = readTDMS(path)
        return data

    def _load(self, start, end):
        """Load data from start to end"""
        data = {}
        dt = start
        while dt < end:
            data_temp = self._load_single(dt)
            if data_temp is None:
                return None 
            if len(data) == 0:
                data = data_temp
            else:
                data= append_dict(data,data_temp)
            dt += self.time_step

        return data
     
    def get_data(self, start: Union[datetime, str], end: Union[datetime, str, None] = None):
        """ Get data from start to end if end is None then load only one signal from start
        Parameters
        ----------
        start : Union[datetime, str]
            start date time
        end : Union[datetime, str] 
            end date time if None then load only one signal from start
        Returns
        -------
        data : np.ndarray
            data from start to end
            """

        if isinstance(start, str):
            start = pd.to_datetime(start)
        if end is None:
            return self._load_single(start)
        if isinstance(end, str):
            end = pd.to_datetime(end)

        if end < start:
            raise ValueError(f'End date {end} is before start date {start}')
        delta = end - start

        if delta > timedelta(days=1):
            raise ValueError(f'Cannot load more than one day at a time. Got {delta.days} days')

        return self._load(start, end)
    
    
if __name__ == '__main__':
    data_root = Path(settings.dataelia.path['raw'])
    sensor = Sensor(name='ACC', location='MO04', data_type='TDD', format='.tdms')
    loader = DataLoader(data_root=data_root,sensor=sensor)
    data = loader.get_data('2022-04-01 00:00:00')
    print(data)
    