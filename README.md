# Muscle Fatigue Classification System Based on EMG Signals

## Abstract
This research paper aims to develop a muscle fatigue classification system based on surface electromyographic (sEMG) signals. It proposes several classifier implementations and feature extraction methods using data from 21 subjects. The systems faced significant challenges in generalizing to individuals outside the training set, but functional classification models were successfully developed for enrolled participants.


## Extracted Features
Classical AI algorithms rely on extracted features to simplify problem complexity. The features used for muscle fatigue classification include:

- **Mean Absolute Value (MAV):** Represents the average absolute muscle activity.
- **Waveform Length (WL):** Quantifies amplitude variations over time.
- **Root Mean Square (RMS):** Indicates energy levels in sEMG signals.
  
- **Zero Crossing Rate (ZCR):** Indicates how often the signal crosses zero.
- **Slope Sign Changes (SSC):** Counts how many times the signal's slope changes.
  

- **Skewness:** Measures asymmetry in signal amplitude distribution.



- **Hjorth Parameter (Activity):** Variance often used in EMG signal analysis.
  


- **Integrated Square-root EMG (ISEMG):** Measures total muscle activity.


- **Mean Frequency (MNF) and Median Frequency (MDF):** Used for EMG 

- **Variation of Central Frequency (VCF):** Measures the spread of 

- **Wavelet Transform (Morlet and Ricker):** Used for detailed 


## Data Preprocessing and Analysis  and System's pipeline

Next, the sequence of steps followed for working with the data and training the classifier is described.

Initially, normalization was applied to each of the acquisition channels for each subject, so that each signal was brought into the range [-1, 1]. The initial unlabelled dataset was divided into signals from unfatigued muscles for the first 35% of samples and signals from fatigued muscles for the last 35% of samples, while 30% of the samples were eliminated to ensure a good separation and avoid misleading the model. After labeling, the signals were split into 250ms windows with 50% overlap, but a variation without overlap is also anticipated for experiments where the test and validation sets contain data from the same subjects. The choice of window size is based on the assumption of stationarity over that short interval, considering the non-isometric nature of the physical exercise performed.

The windows are then directly passed to functions that compute time-domain parameters. For functions calculating frequency-domain parameters, a Hann filter is first applied, preferred for reducing spectral aliasing. Additionally, the first 13 mel-cepstral coefficients are computed. These features are calculated on each channel of each 250ms signal window and then saved as dictionaries, which are later serialized to persistent storage in ".pkl" format.

Starting from the assumption that the system should generalize and discriminate between classes for individuals not in the dataset (we will refer to them as “system-enrolled persons”), the validation set must consist of signal windows from subjects not previously seen by the model during the weight adjustment process. Experiments showed that the dataset’s variability between subjects was insufficient for training the chosen classification systems, so the data will be split into training and validation sets without considering the subjects they originated from.

Due to the small volume of data, cross-validation is used, splitting the group of 20 individuals into sub-groups of four, with one female and one male participant in each group. After training, the model is saved again in ".pkl" format.

## Classifiers Used
The models are capable of distinguishing between signals from fatigued and rested muscles, but this does not ensure generalization to the same individuals seen during training. Therefore, the best models are proposed to be retrained using cross-validation, this time without splitting the data into validation and testing sets based on this criterion, but completely randomly. PCA analysis tends to confirm that the data presents a significant dependence on the person performing the movement.

In addition to the previous metrics, in this stage, the wavelet transform was introduced as a feature. The experimental results of SVM and MLP under these new conditions, with data from all volunteers involved in the training set, led to the results presented. Direct exploitation of time-domain samples was also attempted, but this led to clear overfitting, and the data was not included in the table.

However, there is a problem: because the windows overlap, there is a very high degree of similarity between the windows and the parameters, which is not bad because it plays the role of data augmentation that facilitates the training process. However, the evaluation is not correct in this scenario because the validation and testing sets are abnormally similar, and the network has been partially trained on data from the training set. Therefore, a new methodology for evaluating the results is proposed, where the windows no longer overlap, and the results are presented. As expected, a decrease in scores on the validation set is observed, as well as on the training set, since there is less data available due to the disappearance of the augmentation given by overlapping windows.

A deep network is also attempted, at the input of which the four signal channels brought into the frequency domain by the wavelet transform are applied, this time non-interpolated. Training was done using the Adam optimizer with a learning rate of 0.0001, and the cross-entropy function was chosen as the cost function, just like with the MLP.

From an architectural point of view, the network consists of a number of convolutional layers that have been varied to optimize the complexity of the network. These layers are followed by an adaptive pooling layer that transforms the feature maps so that they can serve as input to fully connected networks that will provide the prediction at the output, in the form of a single neuron with a sigmoid activation function.

Due to the computational volume associated with the model, cross-validation was abandoned, which was otherwise mandatory only when volunteers were kept separate in the training and validation sets.

| Model                      | Channels | Features           | Training ACC | Avg. Validation | Val1  | Val2  | Val3  | Val4  | Val5  |
|----------------------------|----------|--------------------|--------------|-----------------|-------|-------|-------|-------|-------|
| SVM (RBF)                  | 0, 1, 2, 3| time (3)           | 0.701        | 0.564           | 0.527 | 0.518 | 0.568 | 0.506 | 0.502 |
| SVM (RBF)                  | 0, 1, 3  | time (3)           | 0.674        | 0.505           | 0.503 | 0.503 | 0.530 | 0.514 | 0.528 |
| SVM (RBF)                  | 0, 1, 2, 3| frequency (1)      | 0.760        | 0.549           | 0.545 | 0.566 | 0.666 | 0.537 | 0.501 |
| SVM (RBF)                  | 0, 1, 2  | frequency (1)      | 0.722        | 0.564           | 0.555 | 0.546 | 0.662 | 0.551 | 0.478 |
| SVM (RBF)                  | 0, 1, 2, 3| MFCC 13 (3)        | 0.613        | 0.550           | 0.566 | 0.547 | 0.552 | 0.547 | 0.538 |
| SVM (RBF)                  | 0, 1, 2, 3| All features (4)   | 0.642        | 0.548           | 0.497 | 0.514 | 0.646 | 0.561 | 0.522 |

_Table: SVM performance for various features and channels with different validation subjects._


| Model | Features               | Training ACC | Validation | Val1  | Val2  | Val3  | Val4  | Val5  |
|-------|------------------------|--------------|------------|-------|-------|-------|-------|-------|
| MLP   | frequency (1)           | 0.799        | 0.577      | 0.528 | 0.524 | 0.530 | 0.784 | 0.530 |
| MLP   | frequency (1) + MFCC (2)| 0.876        | 0.546      | 0.548 | 0.548 | 0.567 | 0.528 | 0.538 |
| MLP   | All features (4)        | 0.926        | 0.526      | 0.520 | 0.536 | 0.518 | 0.531 | 0.518 |

_Table: MLP results with different feature combinations._


| Model | Features               | Training ACC | Validation | Val1  | Val2  | Val3  | Val4  | Val5  |
|-------|------------------------|--------------|------------|-------|-------|-------|-------|-------|
| MLP   | frequency (1)           | 0.799        | 0.577      | 0.528 | 0.524 | 0.530 | 0.784 | 0.530 |
| MLP   | frequency (1) + MFCC (2)| 0.876        | 0.546      | 0.548 | 0.548 | 0.567 | 0.528 | 0.538 |
| MLP   | All features (4)        | 0.926        | 0.526      | 0.520 | 0.536 | 0.518 | 0.531 | 0.518 |

_Table: MLP (64, 32, 16) results with various feature combinations._



| Model | Architecture  | Training ACC | Validation | Val1  | Val2  | Val3  | Val4  | Val5  |
|-------|---------------|--------------|------------|-------|-------|-------|-------|-------|
| MLP   | (32, 16, 8)   | 0.820        | 0.541      | 0.588 | 0.515 | 0.569 | 0.487 | 0.547 |
| MLP   | (16, 16, 8, 8)| 0.757        | 0.552      | 0.528 | 0.548 | 0.581 | 0.577 | 0.526 |
| MLP   | (8, 8, 8, 8)  | 0.690        | 0.530      | 0.549 | 0.570 | 0.509 | 0.512 | 0.514 |

_Table: Tuning MLP architecture across different feature sets._

## Deep Learning Methods
A deep network is also attempted, where the input consists of the four channels of the signal brought into the frequency domain using the wavelet transform, this time without interpolation. The training was done using the Adam optimizer with a learning rate of 0.0001, and the cost function used was cross-entropy, similar to the MLP.

Architecturally, the network consists of several convolutional layers, which were varied to optimize the network’s complexity. These layers are followed by an adaptive pooling layer that transforms the feature maps, allowing them to serve as input to fully connected layers that ultimately yield the prediction through a single neuron with a sigmoid activation function.

Due to the computational load of the model, cross-validation was not used, which would have been mandatory only if the volunteers were kept separate in the training and validation sets.

The results of the best network are recorded in the table, with training done on non-overlapping windows.

|       | Architecture                                            | Features                             | Training | Mixed Features Validation | Val1  | Val2  | Val3  | Val4  | Val5  |
|-------|---------------------------------------------------------|--------------------------------------|----------|----------------------------|-------|-------|-------|-------|-------|
| SVM   | RBF C=0.3                                               | all (4)                              | 0.576    | 0.565                      | 0.563 | 0.548 | 0.560 | 0.557 | 0.556 |
| MLP   | (256, 128, 64, 32, 16)                                  | all (4)                              | 0.979    | 0.852                      | 0.846 | 0.852 | 0.851 | 0.852 | 0.858 |
| SVM   | RBF C=0.3                                               | Wavelet Ricker (5)                   | 0.577    | 0.578                      | 0.576 | 0.850 | 0.582 | 0.578 | 0.567 |
| MLP   | (256, 128, 64, 32, 16)                                  | Wavelet Ricker (5)                   | 0.896    | 0.810                      | 0.816 | 0.820 | 0.793 | 0.836 | 0.785 |
| MLP   | (256, 128, 64, 32, 16)                                  | Wavelet Morlet (5)                   | 0.926    | 0.819                      | 0.824 | 0.829 | 0.825 | 0.818 | 0.799 |

_Table: Results of MLP and SVM, using all channels, all frequency features, MFCC and time, or using Mexican Hat Wavelet for mixed data between subjects with 50% overlap._


## Conclusions and Future Developments
In conclusion, the best classification systems developed are based on MLP and are capable of distinguishing between the sEMG signals of rested and fatigued muscles, but only for individuals enrolled in the system. In other words, an application based on these models requires an enrollment phase where the user must undergo a specific exercise for the system to function later. There is a quantitative insufficiency of data regarding the number of participants, preventing the systems from generalizing to individuals outside the training set.

It is also noted that frequency-domain features have greater potential for muscle fatigue classification. However, experimental results show that it is also beneficial to use time-domain parameters, which was surprising.

