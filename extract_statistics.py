'''
This is a script that actually creates the dataset
The raw signal are taken and featres are computed on all the given segments, resulting some numers
'''
import os
import numpy as np

from utils_metrici_time import adapt_signal, mean_absolute_value, zero_crossing_rate, slope_sign_changes, root_mean_square, waveform_length, calculate_skewness, hjorth_activity_variance, isemg
from utils_metrici_freq import apply_window, mfcc, compute_mnf, compute_mdf, compute_cf, compute_vcf, compute_wavelet
list_of_time_functions = [mean_absolute_value, zero_crossing_rate, slope_sign_changes, root_mean_square, waveform_length, calculate_skewness, hjorth_activity_variance, isemg]
#list_of_time_functions = [zero_crossing_rate, slope_sign_changes, root_mean_square, waveform_length, calculate_skewness, hjorth_activity_variance, isemg]

#list_of_freq_functions = [mfcc, compute_mnf, compute_mdf]

list_of_freq_functions = [compute_mnf, compute_mdf, compute_cf, compute_vcf, compute_wavelet]

fs = 512

'''
V1) each window will have parameters extracted and they will be saved IN THE FORM OF DICTIONARIE: keys wil be the name of the parameters
'''

#input_dir = 'dataset/dataset_labeled_divided'
#output_dir = 'dataset/dataset_features_labeled_divided'
input_dir = r'.\dataset\dataset_labeled_divided'
output_dir = r'.\dataset\dataset_labeled_divided_features'

####
# Step 1. Iterate through all dataset and find statistics about each featue: average, variance, min and max value. REASON: we aim to normalize the data using these stats
####
# TIMP
'''
list_of_time_features_averages = []
list_of_time_features_variances = []
list_of_time_features_mins = []
list_of_time_features_maxs = []
for function in list_of_time_functions:
    print(function.__name__)
    values_ch0 = []
    values_ch1 = []
    values_ch2 = []
    values_ch3 = []
    for filename in os.listdir(inpt_dir):
        filepath = os.path.join(inpt_dir, filename)
        signal = np.load(filepath)
        #print(np.min(signal), np.max(signal))
        signal = adapt_signal(signal)
        #print(np.min(signal), np.max(signal))
        #print(signal, type(signal))
        values_ch0.append(function(signal[0]))
        values_ch1.append(function(signal[1]))
        values_ch2.append(function(signal[2]))
        values_ch3.append(function(signal[3]))


    minim_ch0 = min(values_ch0)
    maxim_ch0 = max(values_ch0)
    average_ch0 = sum(values_ch0) / len(values_ch0)
    variance_ch0 = np.var(np.array(values_ch0))

    minim_ch1 = min(values_ch1)
    maxim_ch1 = max(values_ch1)
    average_ch1 = sum(values_ch1) / len(values_ch1)
    variance_ch1 = np.var(np.array(values_ch1))

    minim_ch2 = min(values_ch2)
    maxim_ch2 = max(values_ch2)
    average_ch2 = sum(values_ch2) / len(values_ch2)
    variance_ch2 = np.var(np.array(values_ch2))

    minim_ch3 = min(values_ch3)
    maxim_ch3 = max(values_ch3)
    average_ch3 = sum(values_ch3) / len(values_ch3)
    variance_ch3 = np.var(np.array(values_ch3))

    
    print('minim ch0', minim_ch0)
    print('maxim ch0', maxim_ch0)
    print('average ch0', average_ch0)
    print('variance ch0', variance_ch0)

    print('minim ch1', minim_ch1)
    print('maxim ch1', maxim_ch1)
    print('average ch1', average_ch1)
    print('variance ch1', variance_ch1)

    print('minim ch2', minim_ch2)
    print('maxim ch2', maxim_ch2)
    print('average ch2', average_ch2)
    print('variance ch2', variance_ch2)

    print('minim ch3', minim_ch3)
    print('maxim ch3', maxim_ch3)
    print('average ch3', average_ch3)
    print('variance ch3', variance_ch3)
'''

# FRECVENTA nn-MFCC

list_of_freq_features_averages = []
list_of_freq_features_variances = []
list_of_freq_features_mins = []
list_of_freq_features_maxs = []
for function in list_of_freq_functions:
    print(function.__name__)
    values_ch0 = []
    values_ch1 = []
    values_ch2 = []
    values_ch3 = []
    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        signal = np.load(filepath)
        #print(np.min(signal), np.max(signal))
        signal = adapt_signal(signal)

        window = np.hanning(len(signal[0]))
        for i in range(4):
            signal[i] = apply_window(signal[i], window)
        #print(np.min(signal), np.max(signal))
        #print(signal, type(signal))
        values_ch0.append(function(signal[0], fs))
        values_ch1.append(function(signal[1], fs))
        values_ch2.append(function(signal[2], fs))
        values_ch3.append(function(signal[3], fs))


    minim_ch0 = np.min(values_ch0)
    maxim_ch0 = np.max(values_ch0)
    average_ch0 = sum(values_ch0) / len(values_ch0)
    variance_ch0 = np.var(np.array(values_ch0))

    minim_ch1 = np.min(values_ch1)
    maxim_ch1 = np.max(values_ch1)
    average_ch1 = sum(values_ch1) / len(values_ch1)
    variance_ch1 = np.var(np.array(values_ch1))

    minim_ch2 = np.min(values_ch2)
    maxim_ch2 = np.max(values_ch2)
    average_ch2 = sum(values_ch2) / len(values_ch2)
    variance_ch2 = np.var(np.array(values_ch2))

    minim_ch3 = np.min(values_ch3)
    maxim_ch3 = np.max(values_ch3)
    average_ch3 = sum(values_ch3) / len(values_ch3)
    variance_ch3 = np.var(np.array(values_ch3))

    
    print('minim ch0', minim_ch0)
    print('maxim ch0', maxim_ch0)
    print('average ch0', average_ch0)
    print('variance ch0', variance_ch0)

    print('minim ch1', minim_ch1)
    print('maxim ch1', maxim_ch1)
    print('average ch1', average_ch1)
    print('variance ch1', variance_ch1)

    print('minim ch2', minim_ch2)
    print('maxim ch2', maxim_ch2)
    print('average ch2', average_ch2)
    print('variance ch2', variance_ch2)

    print('minim ch3', minim_ch3)
    print('maxim ch3', maxim_ch3)
    print('average ch3', average_ch3)
    print('variance ch3', variance_ch3)


# FRECVENTA MFCC
'''
list_of_freq_features_averages = []
list_of_freq_features_variances = []
list_of_freq_features_mins = []
list_of_freq_features_maxs = []

values_ch0 = []
values_ch1 = []
values_ch2 = []
values_ch3 = []
for filename in os.listdir(inpt_dir):
    filepath = os.path.join(inpt_dir, filename)
    signal = np.load(filepath)
    #print(np.min(signal), np.max(signal))
    signal = adapt_signal(signal)

    window = np.hanning(len(signal[0]))
    for i in range(4):
        signal[i] = apply_window(signal[i], window)
    #print(np.min(signal), np.max(signal))
    #print(signal, type(signal))
    values_ch0.append(mfcc(signal[0], fs))
    values_ch1.append(mfcc(signal[1], fs))
    values_ch2.append(mfcc(signal[2], fs))
    values_ch3.append(mfcc(signal[3], fs))
    for index in range(13):
        print('')
        print('mfcc ', index)

        print(len(values_ch0[0]))

        minim_ch0 = min(values_ch0[index])
        maxim_ch0 = max(values_ch0[index])
        average_ch0 = sum(values_ch0[index]) / len(values_ch0[index])
        variance_ch0 = np.var(np.array(values_ch0[index]))

        minim_ch1 = min(values_ch1[index])
        maxim_ch1 = max(values_ch1[index])
        average_ch1 = sum(values_ch1[index]) / len(values_ch1[index])
        variance_ch1 = np.var(np.array(values_ch1[index]))

        minim_ch2 = min(values_ch2[index])
        maxim_ch2 = max(values_ch2[index])
        average_ch2 = sum(values_ch2[index]) / len(values_ch2[index])
        variance_ch2 = np.var(np.array(values_ch2[index]))

        minim_ch3 = min(values_ch3[index])
        maxim_ch3 = max(values_ch3[index])
        average_ch3 = sum(values_ch3[index]) / len(values_ch3[index])
        variance_ch3 = np.var(np.array(values_ch3[index]))

        
        print('minim ch0', minim_ch0)
        print('maxim ch0', maxim_ch0)
        print('average ch0', average_ch0)
        print('variance ch0', variance_ch0)

        print('minim ch1', minim_ch1)
        print('maxim ch1', maxim_ch1)
        print('average ch1', average_ch1)
        print('variance ch1', variance_ch1)

        print('minim ch2', minim_ch2)
        print('maxim ch2', maxim_ch2)
        print('average ch2', average_ch2)
        print('variance ch2', variance_ch2)

        print('minim ch3', minim_ch3)
        print('maxim ch3', maxim_ch3)
        print('average ch3', average_ch3)
        print('variance ch3', variance_ch3)
'''
'''
list_of_freq_features_averages = []
list_of_freq_features_variances = []
list_of_freq_features_mins = []
list_of_freq_features_maxs = []
for index in range(13):
    print('')
    print('mfcc ', index)    
    values_ch0 = []
    values_ch1 = []
    values_ch2 = []
    values_ch3 = []
    for filename in os.listdir(inpt_dir):
        filepath = os.path.join(inpt_dir, filename)
        signal = np.load(filepath)
        #print(np.min(signal), np.max(signal))
        signal = adapt_signal(signal)

        window = np.hanning(len(signal[0]))
        for i in range(4):
            signal[i] = apply_window(signal[i], window)
        #print(np.min(signal), np.max(signal))
        #print(signal, type(signal))
        values_ch0.append(mfcc(signal[0], fs)[index])
        values_ch1.append(mfcc(signal[1], fs)[index])
        values_ch2.append(mfcc(signal[2], fs)[index])
        values_ch3.append(mfcc(signal[3], fs)[index])


    minim_ch0 = min(values_ch0)
    maxim_ch0 = max(values_ch0)
    average_ch0 = sum(values_ch0) / len(values_ch0)
    variance_ch0 = np.var(np.array(values_ch0))

    minim_ch1 = min(values_ch1)
    maxim_ch1 = max(values_ch1)
    average_ch1 = sum(values_ch1) / len(values_ch1)
    variance_ch1 = np.var(np.array(values_ch1))

    minim_ch2 = min(values_ch2)
    maxim_ch2 = max(values_ch2)
    average_ch2 = sum(values_ch2) / len(values_ch2)
    variance_ch2 = np.var(np.array(values_ch2))

    minim_ch3 = min(values_ch3)
    maxim_ch3 = max(values_ch3)
    average_ch3 = sum(values_ch3) / len(values_ch3)
    variance_ch3 = np.var(np.array(values_ch3))

    
    print('minim ch0', minim_ch0)
    print('maxim ch0', maxim_ch0)
    print('average ch0', average_ch0)
    print('variance ch0', variance_ch0)

    print('minim ch1', minim_ch1)
    print('maxim ch1', maxim_ch1)
    print('average ch1', average_ch1)
    print('variance ch1', variance_ch1)

    print('minim ch2', minim_ch2)
    print('maxim ch2', maxim_ch2)
    print('average ch2', average_ch2)
    print('variance ch2', variance_ch2)

    print('minim ch3', minim_ch3)
    print('maxim ch3', maxim_ch3)
    print('average ch3', average_ch3)
    print('variance ch3', variance_ch3)
    '''



