from load_data import loadData
import os
import numpy as np
import matplotlib.pyplot as plt


data_dir = r'dataset\dataset_cleaned'
output_dir = r'dataset\NORMALIZED_DATASETS\dataset_normalized'

for filename in os.listdir(data_dir):

    input_filepath = os.path.join(data_dir, filename)
    output_filepath = os.path.join(output_dir, filename)
    signal = np.load(input_filepath)

    signal = np.array(signal, dtype=np.float32)
    print(min(signal[0]), max(signal[0]), min(signal[1]), max(signal[1]), min(signal[2]), max(signal[2]), min(signal[3]), max(signal[3]))
    for ch in range(4):
        #signal[ch, :] = signal[ch, :].astype(np.float32) / (max([abs(min(signal[ch, :])), abs(max(signal[ch, :]))]))
        signal[ch, :] = 2 * ( ( (signal[ch, :] - min(signal[ch, :])) / (max(signal[ch, :]) - min(signal[ch, :])) ) -0.5)
    print(min(signal[0]), max(signal[0]), min(signal[1]), max(signal[1]), min(signal[2]), max(signal[2]), min(signal[3]), max(signal[3]))
    
    np.save(output_filepath, signal)