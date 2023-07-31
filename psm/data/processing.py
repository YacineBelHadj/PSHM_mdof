from scipy import signal
import numpy as np

def preprocess_vibration_data(data, filter_order, lpf, sampling_frequency):
    # Step 1: Band-pass filtering
    # Design a Butterworth low-pass filter
    b, a = signal.butter(filter_order, lpf, 'low', fs=sampling_frequency)
    rms= np.sqrt(np.mean(np.square(data)))

    # Apply the filter to the data
    data_filtered = signal.lfilter(b, a, data)
    

    # Step 3: Signal conditioning
    # Subtract the mean and divide by the standard deviation
    conditioned_data = (data_filtered - np.mean(data_filtered)) / np.std(data_filtered)
    return conditioned_data, rms

def apply_welch(sig, sr:int,nperseg:int,noverlap:int|None=None):
    if noverlap is None:
        noverlap = nperseg // 2
    f, Pxx_den = signal.welch(sig, sr, nperseg=nperseg, noverlap=noverlap)
    return f, Pxx_den

def noise_std(SNR_dB:int,RMS_base:float=90):
    SNR_lin = 10**(SNR_dB/20)
    RMS_noise = RMS_base/SNR_lin
    std_noise = RMS_noise
    return std_noise

def generate_noise(SNR_dB:int,signal_length:int,RMS_base:float = 90):
    std_noise = noise_std(SNR_dB,RMS_base=RMS_base)
    noise = np.random.normal(0,std_noise,signal_length)
    return noise

def add_noise(signal,SNR_dB:int,RMS_base:float=90):
    if SNR_dB == 'None':
        return signal
    noise = generate_noise(SNR_dB,signal_length=len(signal),RMS_base=RMS_base)
    return signal + noise

def cut_psd(freq, psd, freq_min, freq_max):
    mask = (freq > freq_min) & (freq < freq_max)
    return freq[mask], psd[mask]
