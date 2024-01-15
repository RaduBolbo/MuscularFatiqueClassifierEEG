from scipy import signal
import matplotlib.pyplot as plt
import numpy as np


####
# Compute spectrogram for just one singl
###
# 
fs = 512

# the data chunk is acually the signal in time o which the FFT will be computed. Thiswill be fet to the DNCNN
data_chuck_size_ms = 2000
data_chuck_size_samples = int(data_chuck_size_ms * fs/1000)
window_overlap_ms = 50 # V1
#window_overlap_ms = 100 # V2
#window_overlap_ms = 256 # V3
window_overlap_samples = int(window_overlap_ms * fs/1000)
window_length_ms = 100 # V1
#window_length_ms = 250 # V2
#window_length_ms = 512 # V3
window_length_samples = int(window_length_ms * fs/1000)
signal_path = r'.\dataset\dataset_labeled\Andrei_Costin_3_r_label=0.npy'
data = np.load(signal_path)[:, 0:data_chuck_size_samples]

# Create a Hann window
window_han = np.hanning(window_length_samples)


# SPECTRUL
for chanel_idx in range(4):
    # Compute the spectrogram
    frequencies, times, Sxx = signal.spectrogram(data[chanel_idx,:], fs=fs, window=window_han, noverlap=window_overlap_samples)
    print(frequencies.shape)
    print(times.shape)
    print(Sxx.shape)

    # mai adaug cate un punct extrapolat pt a corespunde cerintele shadin='flat'
    times = np.append(times, times[-1] + (times[-1] - times[-2]))
    frequencies = np.append(frequencies, frequencies[-1] + (frequencies[-1] - frequencies[-2]))

    # Plotting
    plt.figure(figsize=(10, 6))
    #plt.pcolormesh(times, frequencies, 10*np.log10(Sxx), shading='flat')
    #plt.pcolormesh(times, frequencies, np.log10(Sxx), shading='flat')
    plt.pcolormesh(times, frequencies, Sxx, shading='flat')
    plt.colorbar().set_label('Intensity (dB)')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('Spectrogram with Hamming Window')
    plt.show()

