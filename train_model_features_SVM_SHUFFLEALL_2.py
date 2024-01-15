'''
The "SHUFFLEALL" files don't take into consideration that the test and train data should be from different users.
'''
import os
import pickle
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from sklearn.model_selection import KFold


####
# Combinatii consacrate
####
#feature_names_list = ['hjorthact', 'wl', 'isemg_var', 'zcr'] # doar parametrii in timp relevanti %TESTAT%
feature_names_list = ['mnf', 'mdf', 'cf', 'vcf'] # param de frecveta simpli
#feature_names_list = ['mfc0', 'mfc1', 'mfc2', 'mfc3', 'mfc4', 'mfc5', 'mfc6', 'mfc7', 'mfc8', 'mfc9','mfc10','mfc11', 'mfc12'] # MFCC doar
#feature_names_list = ['mfc0', 'mfc1', 'mfc2', 'mfc3', 'mfc4', 'mfc5', 'mfc6', 'mfc7', 'mfc8', 'mfc9','mfc10','mfc11', 'mfc12', 'mnf', 'mdf', 'cf', 'vcf'] # MFCC + siple freq params
#feature_names_list = [] # WAVELET. Urmeaza
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
        feature_ch = [
            data.get(feature_name + '_ch' + str(ch))[0] if isinstance(data.get(feature_name + '_ch' + str(ch)), np.ndarray) else data.get(feature_name + '_ch' + str(ch)) 
            for feature_name in feature_names_list
            ]
        feature_list += feature_ch

    # Extract the specific features
    return feature_list

#data_directory = './dataset/dataset_labeled_divided_features' # no CF VCF and MFCC saved wrong
#data_directory = './dataset/dataset_labeled_divided_features_2' # added CF VCF and MFCC saved correctly
data_directory = r'./dataset\dataset_labeled_divided_features_3' # added CF VCF and MFCC saved correctly

##################################################################################################
####
# De aici incolo se decide distributia train-cal
####

# se aleg cate 3 baieti si o fata pt fiecare fold

feature_vectors = []
labels = []

for filename in tqdm(os.listdir(data_directory)):
    if filename.endswith('.pkl'):
        flag = False


        file_path = os.path.join(data_directory, filename)
        features = load_features_from_file(file_path)
        feature_vectors.append(features)
        label = int(filename.split('_label=')[1].split('_')[0])
        labels.append(label)




X, y = shuffle(feature_vectors, labels)

X = np.array(X)
y = np.array(y)

#print(X_train)

####
# Train
####

# 1. INSTANTIATE
svm_model = SVC(kernel='linear')
#svm_model = SVC(kernel='rbf', C=0.3)
#svm_model = SVC(kernel='rbf', tol=1e-5)

# 2. TRAIN

kf = KFold(n_splits=5)

train_scores = []
test_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    svm_model.fit(X_train, y_train)
    
    # Evaluate on training set
    y_train_pred = svm_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_scores.append(train_accuracy)
    
    # Evaluate on testing set
    y_test_pred = svm_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_scores.append(test_accuracy)

# 3. EVAL

print("Training scores from each fold: ", train_scores)
print("Average training score: ", sum(train_scores) / len(train_scores))
print("Test scores from each fold: ", test_scores)
print("Average test score: ", sum(test_scores) / len(test_scores))

