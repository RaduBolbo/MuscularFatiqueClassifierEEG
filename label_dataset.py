from load_data import loadData
import os
import numpy as np
import matplotlib.pyplot as plt


#data_dir = 'dataset\dataset_cleaned'
#output_dir = 'dataset\dataset_labeled'
data_dir = r'dataset\NORMALIZED_DATASETS\dataset_normalized'
output_dir = r'dataset\NORMALIZED_DATASETS\dataset_normalized_labeled'
loaddata = loadData(data_dir)

dataStore, labels, filenames = loaddata.loadData_twoClasses_leg_NONORMALIZE(threshold_value=0.5, threshold_width=0.3)
#print(dataStore, labels)
#print(len(dataStore), len(labels))

for data, label, filename in zip(dataStore, labels, filenames):
    #print(filename, label)
    print(min(data[0]), max(data[0]), min(data[1]), max(data[1]), min(data[2]), max(data[2]), min(data[3]), max(data[3]))
    output_filename = filename.split('.')[0] + '_label=' + str(label) + '.npy'
    output_filepath = os.path.join(output_dir, output_filename)
    np.save(output_filepath, data)








