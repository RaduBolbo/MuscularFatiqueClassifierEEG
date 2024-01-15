import os
import torch
from torch.utils.data import Dataset, DataLoader
#import torchaudio.transforms as T
from utils_metrici_freq import wavelet_transform
import numpy as np


class EMGDataset(Dataset):
    def __init__(self, emg_dir):
        self.wavs_dir = emg_dir
        self.files = os.listdir(emg_dir)
        self.wavelet_transform = wavelet_transform
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        emg_path = os.path.join(self.wavs_dir, file_name)
        waveform = np.load(emg_path)
        wavelet = np.zeros((4, 31, 128))
        
        for ch in range(4):
            wavelet[ch, :, :] = self.wavelet_transform(waveform[ch, :], False)
        target = int(file_name.split('_label=')[1].split('_')[0])
        return wavelet, target

# Usage example
# train_dataset = AudioDataset('train_wavs/', 'train_GT_embeddings')
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)