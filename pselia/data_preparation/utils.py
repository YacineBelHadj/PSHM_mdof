from typing import Union
from pathlib import Path
from nptdms import TdmsFile
from datetime import datetime
import warnings
import numpy as np

def readTDMS(path:Union[Path,str]):
    """ Read TDMS file
    Parameters
    ----------
    dt : Union[str,datetime]
        date time of the file to be read
    
    Returns
    -------
        signals : np.ndarray 
            signals from the file
        channels : list
            list of channels in the file
        """
    if isinstance(path, Path):
        path = str(path)

    try:
        data = TdmsFile(path)
    except FileNotFoundError as fnf_error:
            warnings.warn('FAILED IMPORT: No TDMS file at: '+path, UserWarning)
            signals = []
            return signals
    except ValueError as val_error:
            warnings.warn('FAILED IMPORT: TDMS file at: '+path+' seems corrupted. Failed to import.', UserWarning)
            signals = []
            return signals
            
    if len(data.groups()) == 0:
        warnings.warn('FAILED IMPORT: No TDMS group found in file: '+path, UserWarning)
        return None

    signals = [channel.data for channel in data.groups()[0].channels()]
    channels = [channel.name for group in data.groups() for channel in group.channels()]

    dict_signal = dict(zip(channels,signals))

    return dict_signal

def append_dict(dict1:dict, dict2:dict):
    """Append two dictionaries
    Parameters
    ----------
    dict1 : dict
        first dictionary
    dict2 : dict
        second dictionary
    Returns
    -------
    dict1 : dict
        appended dictionary
    """
    for key, value in dict2.items():
        dict1[key]= np.append(dict1[key],value)
    return dict1

def datetime_to_path(path_root:str, dt:datetime,file_type='tdms'):
        """Convert datetime to path
        Parameters
        ----------
        dt : datetime
            datetime to be converted to path
        Returns
        -------
        path : Path
            path of the file
    """
        path = path_root.joinpath(dt.\
            strftime('%Y/%m/%d/%Y%m%d_%H%M%S')).with_suffix(file_type)
        return path
