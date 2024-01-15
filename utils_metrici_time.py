'''
This is a file containing various fucntions that can be applied to the given signal slice for FEATURE EXTRATION
FEATURE EXTRACTION
'''
import librosa
import numpy as np
import os
from scipy.stats import skew
import matplotlib.pyplot as plt
sr = 512

def adapt_signal(signal):
    signal = signal.astype(np.float32) / 128
    return signal
'''
def mfcc(signal, number_of_coeffincients):
    #print(np.max(signal), np.min(signal), type(signal[0]))
    signal = np.array(signal, dtype=float)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=number_of_coeffincients)
    return mfccs
'''
def mean_absolute_value(signal):
    return np.mean(np.abs(signal))

def zero_crossing_rate(signal):
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    return len(zero_crossings) / len(signal)

def slope_sign_changes(signal):
    slope_changes = 0
    for i in range(1, len(signal)-1):
        if (signal[i] - signal[i-1]) * (signal[i] - signal[i+1]) > 0:
            slope_changes += 1
    return slope_changes / len(signal)

def root_mean_square(signal):
    return np.sqrt(np.mean(signal**2))

def waveform_length(signal):
    return np.sum(np.abs(np.diff(signal)))

def calculate_skewness(signal):
    return skew(signal)

def hjorth_activity_variance(signal):
    return np.var(signal)

def isemg(signal):
    abs_emg = np.abs(signal)
    sqrt_emg = np.sqrt(abs_emg)
    isemg = np.sum(sqrt_emg)
    return isemg

if __name__ == '__main__':
    #input_dir = r'evolutie_parametrii_in_timp\divided_window_250ms_overlap_125ms'
    input_dir = r'evolutie_parametrii_in_timp\divided_window_500ms_overlap_250ms'
    #input_dir = r'evolutie_parametrii_in_timp\divided_window_750ms_overlap_375ms' 
    #input_dir = r'evolutie_parametrii_in_timp\divided_window_100ms_overlap_50ms'
    first_mfcc = []
    second_mfcc = []
    mae_list = []
    zcr_list = []
    ssc_list = []
    rms_list = []
    wl_list = []
    skew_list = []
    hjorth_activity_variance_list = []
    isemg_list = []

    files = os.listdir(input_dir)
    files.sort()

    for chanel_idx in range(4):
        for filename in files:
            filepath = os.path.join(input_dir, filename)
            signal = np.load(os.path.join(input_dir, filename))[chanel_idx, :]
            #print(signal.shape)
            signal = adapt_signal(signal)
            #m1, m2 = mfcc(signal=signal, number_of_coeffincients=2)
            #first_mfcc.append(m1)
            #second_mfcc.append(m2)
            mae_list.append(mean_absolute_value(signal))
            zcr_list.append(zero_crossing_rate(signal))
            ssc_list.append(slope_sign_changes(signal))
            rms_list.append(root_mean_square(signal))
            wl_list.append(waveform_length(signal))
            skew_list.append(calculate_skewness(signal))
            hjorth_activity_variance_list.append(hjorth_activity_variance(signal))
            isemg_list.append(isemg(signal))
        # Create a figure and a set of subplots
        fig, axs = plt.subplots(9, 1, figsize=(9, 20))

        

        # features_list = [mae_list, zcr_list, ssc_list, rms_list, wl_list, skew_list, hjorth_activity_variance_list, isemg_list]
        # for feature_list in features_list:
        #     support_time_vector = np.arange(1, len(feature_list) + 1)
        #     correlation = np.corrcoef(feature_list, support_time_vector)[0, 1]
        #     print(correlation)



        axs[0].plot(mae_list, label='MAE')
        axs[0].legend()

        axs[1].plot(zcr_list, label='ZCR')
        axs[1].legend()

        axs[2].plot(ssc_list, label='SSC')
        axs[2].legend()

        axs[3].plot(rms_list, label='RMS')
        axs[3].legend()

        axs[4].plot(wl_list, label='Waveform Length')
        axs[4].legend()

        axs[5].plot(skew_list, label='Skewness')
        axs[5].legend()

        axs[6].plot(hjorth_activity_variance_list, label='Variance(Hjorth activity)')
        axs[6].legend()

        axs[7].plot(isemg_list, label='ISEMG')
        axs[7].legend()

        # Enhance layout
        #plt.tight_layout()
        plt.show()

