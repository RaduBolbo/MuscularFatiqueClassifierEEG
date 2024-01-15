import numpy as np
import os
from tqdm import tqdm


def divide_signal(signal, window_length_ms, wndow_overlap_ms, sr=512):
    window_length_samples = int(window_length_ms * sr/1000)
    wndow_overlap_samples = int(wndow_overlap_ms * sr/1000)
    data = []
    current_index = 0
    
    # comment, caci normalizarea deja s-a facut pe tot semnalul!
    """
    signal = np.array(signal, dtype=np.float32)
    print(min(signal[0]), max(signal[0]), min(signal[1]), max(signal[1]), min(signal[2]), max(signal[2]), min(signal[3]), max(signal[3]))
    for ch in range(4):
        #signal[ch, :] = signal[ch, :].astype(np.float32) / (max([abs(min(signal[ch, :])), abs(max(signal[ch, :]))]))
        signal[ch, :] = 2 * ( ( (signal[ch, :] - min(signal[ch, :])) / (max(signal[ch, :]) - min(signal[ch, :])) ) -0.5)
    print(min(signal[0]), max(signal[0]), min(signal[1]), max(signal[1]), min(signal[2]), max(signal[2]), min(signal[3]), max(signal[3]))
    """
    print('aaaa')
    print(min(signal[0]), max(signal[0]), min(signal[1]), max(signal[1]), min(signal[2]), max(signal[2]), min(signal[3]), max(signal[3]))
    #exit()
    while current_index < signal.shape[1] - window_length_samples:
        data.append(signal[:, current_index : current_index + window_length_samples])
        current_index += window_length_samples - wndow_overlap_samples
    return np.array(data)

####
# Divide al dataset
####

#input_dir = r'E:\an_5_sem1\an_5_sem1\TB\Lab_TB\cod\dataset\dataset_labeled'
#output_dir = r'E:\an_5_sem1\an_5_sem1\TB\Lab_TB\cod\dataset\dataset_labeled_divided_normalized'
#input_dir = r'.\dataset\NORMALIZED_DATASETS\dataset_normalized_labeled'
#output_dir = r'.\dataset\NORMALIZED_DATASETS\dataset_normalized_labeled_divided'
input_dir = r'.\dataset\dataset_labeled'
output_dir = r'.\dataset\NON_OVERLAPING_DATASET\dataset_labeled_divided_nonoveralped'
# Nota: La acest nivel NU este impoortant sa existe o ASOCIERE INTRE FEREASTRA SI OVERLAP - de asta tin cot doar la calculul SPECTROGRAMEI 
# La noi, recomand un overlap lung ca sa fie ca un fel de semi-augumentare a datelor
window_length_ms= 250
wndow_overlap_ms = 125 # aleg overlap mare in speranta ca asta va ajuta ca un fel de augmentare

for filename in tqdm(os.listdir(input_dir)):
    input_filepath = os.path.join(input_dir, filename)
    data = np.load(input_filepath)

    #print(np.all((data >= 0) & (data <= 255) & (data == data.astype(int))))
    data_divided = divide_signal(data, window_length_ms, wndow_overlap_ms)
    for window_idx in range(data_divided.shape[0]):
        output_filepath = os.path.join(output_dir, filename.split('.')[-2] + '_windono_' + str(window_idx) + '.npy')
        np.save(output_filepath, data_divided[window_idx])





####
# Divide one given signal into multiple files
####
'''
#data = np.load(r'evolutie_parametrii_in_timp\Bolborici_Radu_3_r.npy')
#output_dir = r'evolutie_parametrii_in_timp\divided_window_100ms_overlap_50ms'
data = np.load(r'evolutie_parametrii_in_timp\Bolborici_Radu_3_r.npy')
output_dir = r'evolutie_parametrii_in_timp\divided_window_100ms_overlap_50ms'
print(data.shape)
divided_data = divide_signal(data, 100, 50)
print(divided_data.shape)
for idx in range(divided_data.shape[0]):
    savepath = os.path.join(output_dir, str(idx)+'.npy')
    np.save(savepath, divided_data[idx, :, :])
'''



 