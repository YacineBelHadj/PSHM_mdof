from scipy import signal
import numpy as np


def preprocess_vibration_data(data, filter_order, lpf, sampling_frequency):
    if data.ndim == 1:
        data = data[np.newaxis, :]
    # Step 1: Band-pass filtering
    # Design a Butterworth low-pass filter
    data_ = data.copy()
    # correct formula for rms

    b, a = signal.butter(filter_order, lpf, 'low', fs=sampling_frequency)

    # Apply the filter to the data
    data_filtered = signal.filtfilt(b, a, data_, axis=-1)
    

    # Step 3: Signal conditioning
    # Subtract the mean and divide by the standard deviation
# Calculate the mean and standard deviation along the time axis (last axis)
    means = np.mean(data_filtered, axis=-1, keepdims=True)
    stds = np.std(data_filtered, axis=-1, keepdims=True)

    # Condition the data by subtracting the mean and dividing by the standard deviation
    # We keep the mean and std deviations in the same shape as data_filtered for broadcasting
    conditioned_data = (data_filtered - means) / stds
    rms = stds
    return conditioned_data, rms

def apply_welch(sig, sr:int,nperseg:int,noverlap:int|None=None):
    if noverlap is None:
        noverlap = nperseg // 2
    f, Pxx_den = signal.welch(sig, sr, nperseg=nperseg, noverlap=noverlap)
    return f, Pxx_den

def cut_psd(freq, psd, freq_min, freq_max):
    mask = (freq > freq_min) & (freq < freq_max)
    if psd.ndim == 1:
        return freq[mask], psd[mask]
    elif psd.ndim == 2:
        return freq[mask], psd[:, mask]
    
if __name__=='__main__':
    import load_data 
    from pselia.config_elia import settings, load_processed_data_path, load_measurement_bound
    from pathlib import Path
    raw_data_path = Path(settings.dataelia.path['raw'])
    fs = settings.dataelia.sensor['fs']
    params_p = settings.processing['SETTINGS1']

    filter_order = params_p['filter_params']['order']
    lpf = params_p['filter_params']['lpf']
    nperseg = params_p['nperseg']

    sensor = load_data.Sensor(name='ACC', location='MO04', data_type='TDD', format='.tdms')
    loader = load_data.DataLoader(sensor=sensor, data_root=raw_data_path)

    data = loader.get_data(start='2022-04-10 00:00:00',end='2022-04-10 00:10:00')

    sensor_name, signals = zip(*data.items())
    signals = np.array(signals)
    signals_preprocessed ,_= preprocess_vibration_data(signals, filter_order=filter_order, lpf=lpf,sampling_frequency=fs)
    freq, psd = apply_welch(signals_preprocessed,fs,nperseg=nperseg)
    freq_cut , psd_cut = cut_psd(freq ,psd,0,50)
    # psd if we didn't filter the signal
    freq_raw, psd_raw = apply_welch(signals,fs,nperseg=nperseg)
    freq_raw_cut , psd_raw_cut = cut_psd(freq_raw ,psd_raw,0,50)
    # let's plot everything before and after the cut 
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(3,2)
    ax[0,0].plot(signals[3])
    ax[0,0].set_title('raw signal')
    ax[0,1].plot(signals_preprocessed[3])
    ax[0,1].set_title('preprocessed signal')
    ax[1,0].semilogy(freq,psd[3])
    ax[1,0].set_title('psd before cut')
    ax[1,1].semilogy(freq_cut,psd_cut[3])
    ax[1,1].set_title('psd after cut')
    ax[2,0].semilogy(freq_raw,psd_raw[3])
    ax[2,0].set_title('psd before cut raw signal')
    ax[2,1].semilogy(freq_raw_cut,psd_raw_cut[3])
    ax[2,1].set_title('psd after cut raw signal')
    plt.show()
    plt.close()
