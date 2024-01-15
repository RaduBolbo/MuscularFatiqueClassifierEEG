'''
This is a file containing various fucntions that can be applied to the given signal slice for FEATURE EXTRATION
FEATURE EXTRACTION
'''
import librosa
import numpy as np
import os
from scipy.stats import skew
import matplotlib.pyplot as plt
from scipy.signal import welch, periodogram, stft, cwt, ricker, morlet
from scipy.fft import fft, fftfreq
import scipy.interpolate as interpolate
sr = 512



def adapt_signal(signal):
    signal = signal.astype(np.float32) / 128
    return signal

def apply_window(signal, window):
    return signal * window

def mfcc(signal, number_of_coeffincients=13):
    '''
    This assumes that signal is already windowed
    '''
    #print(np.max(signal), np.min(signal), type(signal[0]))
    signal = np.array(signal, dtype=float)
    n_mels = 40
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=number_of_coeffincients, n_fft=128, n_mels=n_mels)
    return mfccs

def compute_psd_forthewholesignal(signal, window_samples, fs): # ATENTIE MARE !!!! LA ASTA calculul este pe tot semnalul, deci nu se asteapta sa primeasca doar o bucata se demnal
    """
    This is used on a large signal not just ona cvasistationary piece of signal.
    """
    f, Pxx = welch(signal, fs=fs, window=window_samples, nperseg=1024, noverlap=512)
    return f, Pxx

def compute_psd_single_segment(windowed_signal, fs):
    '''
    This assumes that signal is already windowed
    ! THis will return an ARRAY rather than a value
    ! Window se aplica o singura data pe tot semnalul. Daca semnalul a fostdeja ferestruit, cel mai bine e sa se dea 'boxcar' sau None
    '''
    f, Pxx = periodogram(windowed_signal, fs=fs)
    return f, Pxx

def intp(cwtmatr, new_length):
    old_length = cwtmatr.shape[1]
    new_index = np.linspace(0, old_length - 1, new_length)
    interpolator = interpolate.interp1d(np.arange(old_length), cwtmatr, kind='linear', axis=1)
    return interpolator(new_index)

def wavelet_transform(windowed_signal, ageraging):
    widths = np.arange(1, 32)
    #cwtmatr = cwt(windowed_signal, ricker, widths)
    cwtmatr = cwt(windowed_signal, morlet, widths)
    if ageraging:
        cwtmatr = intp(cwtmatr, 32)
    return cwtmatr


def compute_mnf(signal, fs):
    """
    This assumes that signal is already windowed.
    """
    n = len(signal)
    fft_vals = fft(signal)
    freqs = fftfreq(n, d=1/fs)

    positive_freqs = freqs[:n//2]
    positive_fft_vals = fft_vals[:n//2]

    power_spectrum = np.abs(positive_fft_vals)**2

    frequency_sum = np.sum(power_spectrum * positive_freqs)
    total_power = np.sum(power_spectrum)
    mnf = frequency_sum / total_power if total_power > 0 else 0

    return mnf

def compute_mdf(windowed_signal, fs):
    fft_vals = fft(windowed_signal)
    n = len(fft_vals)
    
    power_spectrum = np.abs(fft_vals[:n // 2])**2

    cumsum_power_spectrum = np.cumsum(power_spectrum)
    
    total_power = cumsum_power_spectrum[-1]

    mdf_index = np.where(cumsum_power_spectrum >= total_power / 2)[0][0]

    frequencies = np.fft.fftfreq(n, d=1/fs)
    mdf_frequency = frequencies[mdf_index]

    return mdf_frequency

def compute_cf(windowed_signal, fs):
    f, Pxx = compute_psd_single_segment(windowed_signal, fs)
    CF = np.sum(f * Pxx) / np.sum(Pxx) if np.sum(Pxx) != 0 else 0
    return CF

def compute_vcf(windowed_signal, fs):
    f, Pxx = compute_psd_single_segment(windowed_signal, fs)
    
    CF = np.sum(f * Pxx) / np.sum(Pxx) if np.sum(Pxx) != 0 else 0

    VCF = np.sum(((f - CF) ** 2) * Pxx) / np.sum(Pxx) if np.sum(Pxx) != 0 else 0

    return VCF

if __name__ == '__main__':
    #input_dir = r'evolutie_parametrii_in_timp\divided_window_250ms_overlap_125ms'
    input_dir = r'evolutie_parametrii_in_timp\divided_window_500ms_overlap_250ms'
    #input_dir = r'evolutie_parametrii_in_timp\divided_window_750ms_overlap_375ms' 
    #input_dir = r'evolutie_parametrii_in_timp\divided_window_100ms_overlap_50ms'

    fs = 512

    mfcc_0 = []
    mfcc_1 = []
    mfcc_2 = []
    mfcc_3 = []
    mfcc_4 = []
    mnf_list = []
    mdf_list = []
    cf_list = []
    vcf_list = []

    files = os.listdir(input_dir)
    files.sort()

    for chanel_idx in range(4):
        for filename in files:
            filepath = os.path.join(input_dir, filename)
            signal = np.load(os.path.join(input_dir, filename))[chanel_idx, :]
            signal = adapt_signal(signal)
            #print(signal.shape)

            window = np.hanning(len(signal))
            #window = np.hamming(len(signal))
            signal = apply_window(signal, window)

            wavelet = wavelet_transform(signal, ageraging=True)
            print(wavelet.shape)
            
            ###################### ATENTIE! AICI AR TREBUI SA AM GRIJA DE FERESTRUIRE CA SUNT IN DOM FRECAVENTA
            ###################### ATENTIE! AICI AR TREBUI SA AM GRIJA DE FERESTRUIRE 
            #print(signal)
            #print(type(signal))
            #print(signal.shape)
            mfcc_coefs = mfcc(signal=signal, number_of_coeffincients=13)
            #print('llllll')
            #print(type(mfcc_coefs))
            #print(mfcc_coefs)
            mfcc_0.append(mfcc_coefs[0])
            mfcc_1.append(mfcc_coefs[1])
            mfcc_2.append(mfcc_coefs[2])
            mfcc_3.append(mfcc_coefs[3])
            mfcc_4.append(mfcc_coefs[4])
            

            mnf_list.append(compute_mnf(signal, fs))            
            mdf_list.append(compute_mdf(signal, fs))

            cf_list.append(compute_cf(signal, fs))            
            vcf_list.append(compute_vcf(signal, fs))

        # Create a figure and a set of subplots
        fig, axs = plt.subplots(9, 1, figsize=(9, 20))

        # Plot each list
        axs[0].plot(mfcc_0, label='First MFCC')
        axs[0].legend()

        axs[1].plot(mfcc_1, label='Second MFCC')
        axs[1].legend()

        axs[2].plot(mfcc_2, label='3rd MFCC')
        axs[2].legend()

        axs[3].plot(mfcc_3, label='4th MFCC')
        axs[3].legend()

        axs[4].plot(mfcc_4, label='5th MFCC')
        axs[4].legend()

        axs[5].plot(mnf_list, label='MNF')
        axs[5].legend()

        axs[6].plot(mdf_list, label='MDF')
        axs[6].legend()

        axs[7].plot(cf_list, label='CF')
        axs[7].legend()

        axs[8].plot(vcf_list, label='VCF')
        axs[8].legend()
        # Enhance layout
        #plt.tight_layout()
        plt.show()

