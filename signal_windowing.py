import numpy as np
import matplotlib.pyplot as plt


####
# Apply Hann and hamming to just one example
####
signal_path = r'E:\an_5_sem1\TB\Lab_TB\cod\dataset\dataset_labeled_divided\Andrei_Costin_3_r_label=0_windono_28.npy'
signal = np.load(signal_path)[0]

signal_length = signal.shape[0]

# Create a Hann window
window_han = np.hanning(signal_length)
window_hamming = np.hamming(signal_length)

# Apply the window to the signal
windowed_signal_han = signal * window_han
windowed_signal_hamming = signal * window_hamming

# Plotting
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(signal)
plt.title('Original Signal')
plt.subplot(3, 1, 2)
plt.plot(windowed_signal_han)
plt.title('Signal with Hann Window')
plt.subplot(3, 1, 3)
plt.plot(windowed_signal_hamming)
plt.title('Signal with Hamming Window')
plt.tight_layout()
plt.show()




