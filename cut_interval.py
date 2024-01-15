from load_data import loadData
import numpy as np


loaddata = loadData('dataset')

def cut_signal(signal, start_sample, stop_sample):
    cut_signal = signal[0:4, start_sample:stop_sample]

    return cut_signal


#input_file = r'E:\an_5_sem1\TB\Lab_TB\cod\dataset\dataset_original\Dumitru_Marius-Vlad_3_r.npy' # 0 - 40000
#input_file = r'E:\an_5_sem1\TB\Lab_TB\cod\dataset\dataset_original\Neamtu_Cristian_3_r.npy' # 0 - 54000
#input_file = r'E:\an_5_sem1\TB\Lab_TB\cod\dataset\dataset_original\Petrache_Adrian-Alberto_3_r.npy' # 0 - 50000
input_file = r'E:\an_5_sem1\TB\Lab_TB\cod\dataset\dataset_original\Scrieciu_Robert_3_R.npy' # 0 - 40000

#output_file = r'E:\an_5_sem1\TB\Lab_TB\cod\dataset\dataset_cleaned\Dumitru_Marius-Vlad_3_r.npy'
#output_file = r'E:\an_5_sem1\TB\Lab_TB\cod\dataset\dataset_cleaned\Neamtu_Cristian_3_r.npy'
#output_file = r'E:\an_5_sem1\TB\Lab_TB\cod\dataset\dataset_cleaned\Petrache_Adrian-Alberto_3_r.npy'
output_file = r'E:\an_5_sem1\TB\Lab_TB\cod\dataset\dataset_cleaned\Scrieciu_Robert_3_R.npy'

data = loaddata.load_data(filename=input_file)
data_cut = cut_signal(data, 0, 40000)
print(data_cut.shape)

np.save(output_file, data_cut)