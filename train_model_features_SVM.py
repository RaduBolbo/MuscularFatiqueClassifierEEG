import os
import pickle
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

#feature_names_list = ['mav'] # 0.59 0.52
#feature_names_list = ['zcr'] # 0.62 0.52
#feature_names_list = ['rms'] # 0.59 0.52
#feature_names_list = ['isemg_var'] # 0.619 0.52
#feature_names_list = ['wl'] # 0.6 0.53s
#feature_names_list = ['skew'] # 0.56 0.46
#feature_names_list = ['hjorthact'] # 0.56 0.56

#feature_names_list = ['mnf'] # 0.63, 0.51
#feature_names_list = ['mdf'] # 0.60 0.53



#feature_names_list = ['mfcc']

#feature_names_list = ['mav', 'zcr']

####
# Combinatii consacrate
####
#feature_names_list = ['hjorthact', 'wl', 'isemg_var', 'zcr'] # doar parametrii in timp relevanti %TESTAT%
feature_names_list = ['mnf', 'mdf', 'cf', 'vcf'] # param de frecveta simpli
#feature_names_list = ['mfc0', 'mfc1', 'mfc2', 'mfc3', 'mfc4', 'mfc5', 'mfc6', 'mfc7', 'mfc8', 'mfc9','mfc10','mfc11', 'mfc12'] # MFCC doar
#feature_names_list = ['mfc0', 'mfc1', 'mfc2', 'mfc3', 'mfc4', 'mfc5', 'mfc6', 'mfc7', 'mfc8', 'mfc9','mfc10','mfc11', 'mfc12', 'mnf', 'mdf', 'cf', 'vcf'] # MFCC + siple freq params
#feature_names_list = ['WL'] # WAVELET. Urmeaza
#feature_names_list = ['mav', 'rms', 'zcr', 'ssc', 'isemg_var', 'wl', 'skew', 'hjorthact', 'mnf', 'mdf'] #TOTI parametrii de timp (iesise okish)
#feature_names_list = ['mav', 'rms', 'zcr', 'ssc', 'isemg_var', 'wl', 'skew', 'hjorthact', 'mnf', 'mdf', 'cf', 'vcf']
#feature_names_list = ['mav', 'rms', 'zcr', 'ssc', 'isemg_var', 'wl', 'skew', 'hjorthact', 'mnf', 'mdf', 'cf', 'vcf', 'mfc0', 'mfc1', 'mfc2', 'mfc3', 'mfc4', 'mfc5', 'mfc6', 'mfc7', 'mfc8', 'mfc9','mfc10','mfc11', 'mfc12', 'mnf', 'mdf', 'cf', 'vcf'] # toti param de timp + topti param freq + mfcc

#feature_names_list = ['mav', 'rms', 'zcr', 'ssc', 'isemg_var', 'wl', 'skew', 'hjorthact', 'mnf', 'mdf'] # ch 0, 1, 3 -> 0.75 si 0.6 sau 0, 1, 2, 3 -> 0.78 cu 0.64
#used_channels = [0, 1, 3] # integers. 0, 1, 3 e cel mai bine
used_channels = [0, 1, 2, 3] # sau

def load_features_from_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    feature_list = []
    
    for ch in used_channels:
        #feature_ch = [data.get(feature_name + '_ch' + str(ch)) for feature_name in feature_names_list] 
        feature_ch = [
            data.get(feature_name + '_ch' + str(ch))[0] if isinstance(data.get(feature_name + '_ch' + str(ch)), np.ndarray) else data.get(feature_name + '_ch' + str(ch)) 
            for feature_name in feature_names_list
            ]
        #print(feature_ch)
        #print('llllll')
        #print(feature_ch)
        '''
        for i in range(len(feature_ch)):
            if isinstance(feature_ch[i], np.ndarray):
                array = feature_ch.pop(i)
                for elem in array:
                    feature_ch.append(elem)
        '''
        feature_list += feature_ch

    # Extract the specific features
    #print(feature_list) 8
    return feature_list

#data_directory = './dataset/dataset_labeled_divided_features' # no CF VCF and MFCC saved wrong
data_directory = './dataset/dataset_labeled_divided_features_3' # added Wavelet
#data_directory = r'.\dataset\NORMALIZED_DATASETS\dataset_normalized_labeled_divided_features' # added CF VCF and MFCC saved correctly

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
        if filename.endswith('.pkl'):
            flag = False
            for name in val_subjects:
                if filename.startswith(name):
                    flag = True
            if flag:
                file_path = os.path.join(data_directory, filename)
                features = load_features_from_file(file_path)
                val_feature_vectors.append(features)
                label = int(filename.split('_label=')[1].split('_')[0])
                val_labels.append(label)
            else:
                file_path = os.path.join(data_directory, filename)
                features = load_features_from_file(file_path)
                train_feature_vectors.append(features)
                label = int(filename.split('_label=')[1].split('_')[0])
                train_labels.append(label)



    X_train, y_train = shuffle(train_feature_vectors, train_labels)
    #X_train, y_train = train_feature_vectors, train_labels
    X_val, y_val = val_feature_vectors, val_labels

    #print(X_train)

    ####
    # Train
    ####

    # 1. INSTANTIATE
    svm_model = SVC(kernel='linear')
    #svm_model = SVC(kernel='rbf', C=0.3)
    #svm_model = SVC(kernel='rbf', tol=1e-5)

    # 2. TRAIN
    svm_model.fit(X_train, y_train)

    # 3. EVAL

    # On Train data:
    print('CORSS VALIDATION FOLD: ', fold_idx)
    y_pred_train = svm_model.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    print("Accuracy:", accuracy_train)
    print("Classification Report:")
    print(classification_report(y_train, y_pred_train))

    # On Val data
    y_pred_val = svm_model.predict(X_val)
    accuracy_val = accuracy_score(y_val, y_pred_val)
    print("Accuracy:", accuracy_val)
    print("Classification Report:")
    print(classification_report(y_val, y_pred_val))

