'''
This is a script that actually creates the dataset
The raw signal are taken and featres are computed on all the given segments, resulting some numers
'''
import os
import numpy as np
from utils_metrici_time import mean_absolute_value, zero_crossing_rate, slope_sign_changes, root_mean_square, waveform_length, calculate_skewness, hjorth_activity_variance, isemg
from utils_metrici_freq import apply_window, compute_wavelet, mfcc, compute_mnf, compute_mdf,compute_cf, compute_vcf
import pickle
from tqdm import tqdm

list_of_time_functions = [mean_absolute_value, zero_crossing_rate, slope_sign_changes, root_mean_square, waveform_length, calculate_skewness, hjorth_activity_variance, isemg]
#list_of_time_functions = [zero_crossing_rate, slope_sign_changes, root_mean_square, waveform_length, calculate_skewness, hjorth_activity_variance, isemg]

#list_of_freq_functions = [mfcc, compute_mnf, compute_mdf]

list_of_freq_functions = [compute_mnf, compute_mdf, compute_wavelet]

# define the minimum and maximum values for each feature
mav_min = 0
mav_max = 0.9

zcr_min = 0
zcr_max = 0.93

ssc_min = 0.03
ssc_max = 0.92

rms_min = 0
rms_max = 0.9

wl_min = 0.4375
wl_max = 79

skew_min = -10.93
skew_max = 10.7

hjorthact_min = 0
hjorthact_max = 0.84

isemg_min = 4.89
isemg_max = 120

# freq
mnf_min = 1
mnf_max = 188

mdf_min = 0
mdf_max = 228

cf_min = 0.45
cf_max = 205

vcf_min = 13
vcf_max = 13070

# mfcc
mfcc_min = np.array([-345, -34, -49, -44, -37, -37, -32, -27, -34, -29, -22.5, -25.5, -21.3])
mfcc_max = np.array([-43, 104, 55, 46, 39, 35, 34, 34, 33.6, 30, 26.5, 28, 25.5])

#wave
wave_min = -5.64
wave_max = 5.53
inpt_dir = r'.\dataset\dataset_labeled_divided'
output_dir = r'./dataset/dataset_labeled_divided_features_3'
fs = 512

for filename in tqdm(os.listdir(inpt_dir)):
        input_filepath = os.path.join(inpt_dir, filename)
        output_filepath = os.path.join(output_dir, filename[:-4] + '.pkl')

        signal = np.load(input_filepath)
        #print(type(signal))
        #print(signal.shape)
        #print(np.min(signal), np.max(signal))
        #signal_time = adapt_signal(signal)
        signal_time = signal.copy()
        signal_frequency = signal.copy()

        window = np.hanning(len(signal[0]))
        for i in range(4):
            signal_frequency[i, :] = apply_window(signal[i, :], window)
        
        # initializez un dictionar cu trasaturi
        feature_dict = {}    
        # iterez prin canale
        for ch in range(4):
            #print(signal_time[ch, :].shape)
            #print(signal_frequency[ch, :].shape)
            # timp
            mav = (mean_absolute_value(signal_time[ch, :]) - mav_min) / (mav_max - mav_min)
            feature_dict["mav_ch"+str(ch)] = mav
            zcr = (zero_crossing_rate(signal_time[ch, :]) - zcr_min) / (zcr_max - zcr_min)
            feature_dict["zcr_ch"+str(ch)] = zcr
            ssc = (slope_sign_changes(signal_time[ch, :]) - ssc_min) / (ssc_max - ssc_min)
            feature_dict["ssc_ch"+str(ch)] = ssc
            rms = (root_mean_square(signal_time[ch, :]) - rms_min) / (zcr_max - rms_min)
            feature_dict["rms_ch"+str(ch)] = rms
            wl = (waveform_length(signal_time[ch, :]) - wl_min) / (wl_max - wl_min)
            feature_dict["wl_ch"+str(ch)] = wl
            skew = (calculate_skewness(signal_time[ch, :]) - skew_min) / (skew_max - skew_min)
            feature_dict["skew_ch"+str(ch)] = skew
            hjorthact = (hjorth_activity_variance(signal_time[ch, :]) - hjorthact_min) / (hjorthact_max - hjorthact_min)
            feature_dict["hjorthact_ch"+str(ch)] = hjorthact
            isemg_var = (isemg(signal_time[ch, :]) - isemg_min) / (isemg_max - isemg_min)
            feature_dict["isemg_var_ch"+str(ch)] = isemg_var


            # freq no mfcc
            mnf = (compute_mnf(signal_frequency[ch, :], fs) - mnf_min) / (mnf_max - mnf_min)
            feature_dict["mnf_ch"+str(ch)] = mnf 
            mdf = (compute_mdf(signal_frequency[ch, :], fs) - mdf_min) / (mdf_max - mdf_min)
            feature_dict["mdf_ch"+str(ch)] = mdf
            cf_var = (compute_cf(signal_frequency[ch, :], fs) - cf_min) / (cf_max - cf_min)
            feature_dict["cf_ch"+str(ch)] = isemg_var
            vcf_var = (compute_vcf(signal_frequency[ch, :], fs) - vcf_min) / (vcf_max - vcf_min)
            feature_dict["vcf_ch"+str(ch)] = isemg_var

            # freq mfcc
            unnormalized_mfcc = np.array(list(mfcc(signal_frequency[ch], 13)))

            for i in range(13): 
                normalized_mfcc = (unnormalized_mfcc[i] - mfcc_min[i]) / (mfcc_max[i] - mfcc_min[i])
                feature_dict["mfc" + str(i) + "_ch" + str(ch)] = normalized_mfcc
                #print(feature_dict["mfc" + str(i) + "_ch" + str(ch)])

            normalized_WL = (compute_wavelet(signal_time[ch, :], fs) - wave_min) / (wave_max-wave_min)
            feature_dict["WL"+str(i) + "_ch" + str(ch)] = normalized_WL

        #print(feature_dict)

        with open(output_filepath, 'wb') as f:
            pickle.dump(feature_dict, f)



