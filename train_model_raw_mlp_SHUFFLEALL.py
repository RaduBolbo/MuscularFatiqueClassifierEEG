'''
The "SHUFFLEALL" files don't take into consideration that the test and train data should be from different users.
'''
import os
import pickle
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from sklearn.model_selection import KFold



used_channels = [0, 1, 2, 3] # sau

#data_directory = './dataset/dataset_labeled_divided_features' # no CF VCF and MFCC saved wrong
#data_directory = './dataset/dataset_labeled_divided_features_2' # added CF VCF and MFCC saved correctly
#data_directory = r'.\dataset\NORMALIZED_DATASETS\dataset_normalized_labeled_divided_features' # added CF VCF and MFCC saved correctly
data_directory = r'.\dataset\NON_OVERLAPING_DATASET\dataset_labeled_divided_nonoveralped'

##################################################################################################
####
# De aici incolo se decide distributia train-cal
####

# se aleg cate 3 baieti si o fata pt fiecare fold

feature_vectors = []
labels = []


for filename in tqdm(os.listdir(data_directory)):
    if filename.endswith('.npy'):
        file_path = os.path.join(data_directory, filename)
        data = np.load(file_path)

        feature_vector = np.concatenate([data[0], data[1], data[2], data[3]])

        feature_vectors.append(feature_vector)
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
mlp_model = MLPClassifier(hidden_layer_sizes=(32, 16, 8), activation='relu', 
                    max_iter=1000) # learning_rate_init ERA INITIAL 0.001

# 2. TRAIN

kf = KFold(n_splits=5)

train_scores = []
test_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    mlp_model.fit(X_train, y_train)
    
    # Evaluate on training set
    y_train_pred = mlp_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_scores.append(train_accuracy)
    
    # Evaluate on testing set
    y_test_pred = mlp_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_scores.append(test_accuracy)

# 3. EVAL

print("Training scores from each fold: ", train_scores)
print("Average training score: ", sum(train_scores) / len(train_scores))
print("Test scores from each fold: ", test_scores)
print("Average test score: ", sum(test_scores) / len(test_scores))

















