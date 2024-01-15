from load_data import loadData
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_signals(data, label):
    fig, axs = plt.subplots(4, 1, figsize=(10, 20))  # 8 subplots for 8 signals
    for i in range(4):
        axs[i].plot(data[i])
        axs[i].set_title(f'Canal {i+1}')
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

data_dir = 'dataset\dataset_original'
loaddata = loadData(data_dir)

####
# 1) view 8 chanels
####
'''
#data_dir = 'dataset\dataset_cleaned'
for filename in os.listdir(data_dir):
    filepath = os.path.join(data_dir, filename)
    data = loaddata.load_data(filename=filepath)
    print(data.shape)
    print(np.min(data[0]), np.max(data[0]))
    print(np.min(data[1]), np.max(data[1]))
    print(np.min(data[2]), np.max(data[2]))
    print(np.min(data[3]), np.max(data[3]))
    print(np.min(data[4]), np.max(data[4]))
    print(np.min(data[5]), np.max(data[5]))
    print(np.min(data[6]), np.max(data[6]))
    print(np.min(data[7]), np.max(data[7]))
    print(filename)
    plot_signals(data, 'undefined')
'''
####
# 2) view 4 chanels
####

data_dir = 'dataset\dataset_original'
for filename in os.listdir(data_dir):
    filepath = os.path.join(data_dir, filename)
    data = loaddata.load_data(filename=filepath)
    print(data.shape)
    print(np.min(data[0]), np.max(data[0]))
    print(np.min(data[1]), np.max(data[1]))
    print(np.min(data[2]), np.max(data[2]))
    print(np.min(data[3]), np.max(data[3]))
    print(filename)
    plot_signals(data, 'undefined')







