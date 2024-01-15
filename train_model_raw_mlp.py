import os
import pickle
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pywt

#feature_names_list = ['mav', 'rms', 'zcr', 'ssc', 'isemg_var', 'wl', 'skew', 'hjorthact', 'mnf', 'mdf'] # ch 0, 1, 3 -> 0.75 si 0.6 sau 0, 1, 2, 3 -> 0.78 cu 0.64
#used_channels = [0, 1, 3] # integers. 0, 1, 3 e cel mai bine
used_channels = [0, 1, 2, 3] # sau

#data_directory = './dataset/dataset_labeled_divided_features' # no CF VCF and MFCC saved wrong
#data_directory = './dataset/dataset_labeled_divided_features_2' # added CF VCF and MFCC saved correctly
data_directory = r'.\dataset\NORMALIZED_DATASETS\dataset_normalized_labeled_divided' # added CF VCF and MFCC saved correctly

##################################################################################################
####
# De aici incolo se decide distributia train-cal
####

# se aleg cate 3 baieti si o fata pt fiecare fold
val_subjects_cross_validation = [['Andrei_Costin', 'Balabaneanu_Madalina', 'Bolborici_Radu', 'Busuioc_Alexandru'], ['Dobre_Adrian', 'Dumitru_Alexandru', 'Dumitru_Marius-Vlad', 'Guta_Catalina'], ['Ionita_Rebeca', 'Neamtu_Cristian', 'Nuteanu_Dorin', 'Ovidiu_Burcea'], ['Manolache_Malina', 'Petrache_Adrian-Alberto', 'Popa_Cosmin', 'Radu_Vali'], ['Susnea_Maria', 'Saia_Silviu', 'Scrieciu_Robert', 'Vlad_Cristian']]

for fold_idx, val_subjects in enumerate(val_subjects_cross_validation):
    train_feature_vectors = []
    train_labels = []

    val_feature_vectors = []
    val_labels = []

    for filename in tqdm(os.listdir(data_directory)):
        if filename.endswith('.npy'):
            flag = False
            for name in val_subjects:
                if filename.startswith(name):
                    flag = True
            if flag:
                file_path = os.path.join(data_directory, filename)
                data = np.load(file_path)

                feature_vector = np.concatenate([data[0], data[1], data[2], data[3]])

                val_feature_vectors.append(feature_vector)
                label = int(filename.split('_label=')[1].split('_')[0])
                val_labels.append(label)
            else:
                file_path = os.path.join(data_directory, filename)
                data = np.load(file_path)

                feature_vector = np.concatenate([data[0], data[1], data[2], data[3]])

                train_feature_vectors.append(feature_vector)
                label = int(filename.split('_label=')[1].split('_')[0])
                train_labels.append(label)



    X_train, y_train = shuffle(train_feature_vectors, train_labels)
    #X_train, y_train = train_feature_vectors, train_labels
    X_val, y_val = val_feature_vectors, val_labels



    ####
    # Train
    ####

    # 1. INSTANTIATE
    mlp = MLPClassifier(hidden_layer_sizes=(4, 4, 4), activation='relu', 
                    max_iter=1000)

    # hidden_layer_sizes=(100, 100, 100), activation='relu', max_iter=1000, random_state=42 # 0.92 cu 0.52



    # 2. TRAIN
    
    mlp.fit(X_train, y_train)

    # 3. EVAL

    # On Train data:
    print('CORSS VALIDATION FOLD: ', fold_idx)
    y_pred_train = mlp.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    print("Accuracy:", accuracy_train)
    print("Classification Report:")
    print(classification_report(y_train, y_pred_train))

    # On Val data
    y_pred_val = mlp.predict(X_val)
    accuracy_val = accuracy_score(y_val, y_pred_val)
    print("Accuracy:", accuracy_val)
    print("Classification Report:")
    print(classification_report(y_val, y_pred_val))
    

