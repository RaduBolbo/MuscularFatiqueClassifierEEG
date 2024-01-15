import os
import numpy as np


class loadData():

    def __init__(self,dataDirectory):
        self.dataDirectory = dataDirectory

    def load_data(self,filename):
        data = np.load(filename)
        return data

    def loadData_armthreeClasses(self):
            dataStore = []
            labels = []
            for filename in os.listdir(self.dataDirectory):
                if filename.endswith('.npy'):
                    file_data = self.load_data(os.path.join(self.dataDirectory, filename))
                    parts = filename.split('_')
                    print(parts)
                    cl = parts[2]
                    if cl == "0":
                        channel1 = file_data[0].astype(int)-128
                        channel2 = file_data[1].astype(int)-128
                        channel3 = file_data[2].astype(int)-128
                        channel4 = file_data[3].astype(int)-128
                        channel5 = file_data[4].astype(int)-128
                        channel6 = file_data[5].astype(int)-128
                        channel7 = file_data[6].astype(int)-128
                        channel8 = file_data[7].astype(int)-128
                        lis = [channel1, channel2, channel3, channel4, channel5, channel6, channel7, channel8]
                        dataStore.append(lis)
                        labels.append(0)
                    elif cl == "1":
                        channel1 = file_data[0].astype(int)-128
                        channel2 = file_data[1].astype(int)-128
                        channel3 = file_data[2].astype(int)-128
                        channel4 = file_data[3].astype(int)-128
                        channel5 = file_data[4].astype(int)-128
                        channel6 = file_data[5].astype(int)-128
                        channel7 = file_data[6].astype(int)-128
                        channel8 = file_data[7].astype(int)-128
                        lis = [channel1, channel2, channel3, channel4, channel5, channel6, channel7, channel8]
                        dataStore.append(lis)
                        labels.append(1)
                    elif cl == "2":
                        channel1 = file_data[0].astype(int)-128
                        channel2 = file_data[1].astype(int)-128
                        channel3 = file_data[2].astype(int)-128
                        channel4 = file_data[3].astype(int)-128
                        channel5 = file_data[4].astype(int)-128
                        channel6 = file_data[5].astype(int)-128
                        channel7 = file_data[6].astype(int)-128
                        channel8 = file_data[7].astype(int)-128
                        lis = [channel1, channel2, channel3, channel4, channel5, channel6, channel7, channel8]
                        dataStore.append(lis)
                        labels.append(2)
            return dataStore,labels

    def loadData_twoClasses_leg(self, threshold_value=0.5, threshold_width=0):
            dataStore = []
            labels = []
            filenames = []
            for filename in os.listdir(self.dataDirectory):
                if filename.endswith('.npy'):
                    file_data = self.load_data(os.path.join(self.dataDirectory, filename))
                    parts = filename.split('_')
                    print(parts)
                    cl = parts[2]
                    if cl == "3":
                        channel1 = file_data[0].astype(int)-128
                        channel2 = file_data[1].astype(int)-128
                        channel3 = file_data[2].astype(int)-128
                        channel4 = file_data[3].astype(int)-128
                        data_length = len(file_data[0])
                        print(data_length)
                        upper_limit_ODIHNIT = int(data_length*threshold_value - data_length*threshold_width/2)
                        lower_limit_OBOSIT = int(data_length*threshold_value + data_length*threshold_width/2)
                        #lis = [channel1[:20480], channel2[:20480], channel3[:20480], channel4[:20480]]    # 0 to 40s
                        lis = [channel1[:upper_limit_ODIHNIT], channel2[:upper_limit_ODIHNIT], channel3[:upper_limit_ODIHNIT], channel4[:upper_limit_ODIHNIT]]    # 0 to 40s
                        dataStore.append(lis)
                        labels.append(0)
                        #lis = [channel1[20480:40960], channel2[20480:40960], channel3[20480:40960], channel4[20480:40960]]    # 40 to 80s
                        lis = [channel1[lower_limit_OBOSIT:], channel2[lower_limit_OBOSIT:], channel3[lower_limit_OBOSIT:], channel4[lower_limit_OBOSIT:]]    # 40 to 80s
                        dataStore.append(lis)
                        labels.append(1)
                        filenames.append(filename)
                        filenames.append(filename)
            return dataStore, labels, filenames
    
    def loadData_twoClasses_leg_NONORMALIZE(self, threshold_value=0.5, threshold_width=0):
        dataStore = []
        labels = []
        filenames = []
        for filename in os.listdir(self.dataDirectory):
            if filename.endswith('.npy'):
                file_data = self.load_data(os.path.join(self.dataDirectory, filename))
                parts = filename.split('_')
                print(parts)
                cl = parts[2]
                if cl == "3":
                    channel1 = file_data[0].astype(np.float32)
                    channel2 = file_data[1].astype(np.float32)
                    channel3 = file_data[2].astype(np.float32)
                    channel4 = file_data[3].astype(np.float32)
                    data_length = len(file_data[0])
                    print(data_length)
                    upper_limit_ODIHNIT = int(data_length*threshold_value - data_length*threshold_width/2)
                    lower_limit_OBOSIT = int(data_length*threshold_value + data_length*threshold_width/2)
                    #lis = [channel1[:20480], channel2[:20480], channel3[:20480], channel4[:20480]]    # 0 to 40s
                    lis = [channel1[:upper_limit_ODIHNIT], channel2[:upper_limit_ODIHNIT], channel3[:upper_limit_ODIHNIT], channel4[:upper_limit_ODIHNIT]]    # 0 to 40s
                    dataStore.append(lis)
                    labels.append(0)
                    #lis = [channel1[20480:40960], channel2[20480:40960], channel3[20480:40960], channel4[20480:40960]]    # 40 to 80s
                    lis = [channel1[lower_limit_OBOSIT:], channel2[lower_limit_OBOSIT:], channel3[lower_limit_OBOSIT:], channel4[lower_limit_OBOSIT:]]    # 40 to 80s
                    dataStore.append(lis)
                    labels.append(1)
                    filenames.append(filename)
                    filenames.append(filename)
        return dataStore, labels, filenames


    def loadData_twoClasses_firstarmmovement(self):
            dataStore = []
            labels = []
            for filename in os.listdir(self.dataDirectory):
                if filename.endswith('.npy'):
                    file_data = self.load_data(os.path.join(self.dataDirectory, filename))
                    parts = filename.split('_')
                    print(parts)
                    cl = parts[2]
                    if cl == "0":
                        channel1 = file_data[0].astype(int)-128
                        channel2 = file_data[1].astype(int)-128
                        channel3 = file_data[2].astype(int)-128
                        channel4 = file_data[3].astype(int)-128
                        channel5 = file_data[4].astype(int)-128
                        channel6 = file_data[5].astype(int)-128
                        channel7 = file_data[6].astype(int)-128
                        channel8 = file_data[7].astype(int)-128
                        lis = [channel1[:15360], channel2[:15360], channel3[:15360], channel4[:15360], channel5[:15360], channel6[:15360], channel7[:15360], channel8[:15360]]    # 0s to 30s
                        dataStore.append(lis)
                        labels.append(0)
                        lis = [channel1[15360:], channel2[15360:], channel3[15360:], channel4[15360:], channel5[15360:], channel6[15360:], channel7[15360:], channel8[15360:]]    # 30s to 60s
                        dataStore.append(lis)
                        labels.append(1)
            return dataStore,labels

    def loadData_twoClasses_secondarmmovement(self):
        dataStore = []
        labels = []
        for filename in os.listdir(self.dataDirectory):
            if filename.endswith('.npy'):
                file_data = self.load_data(os.path.join(self.dataDirectory, filename))
                parts = filename.split('_')
                print(parts)
                cl = parts[2]
                if cl == "1":
                    channel1 = file_data[0].astype(int)-128
                    channel2 = file_data[1].astype(int)-128
                    channel3 = file_data[2].astype(int)-128
                    channel4 = file_data[3].astype(int)-128
                    channel5 = file_data[4].astype(int)-128
                    channel6 = file_data[5].astype(int)-128
                    channel7 = file_data[6].astype(int)-128
                    channel8 = file_data[7].astype(int)-128
                    lis = [channel1[5120:15360], channel2[5120:15360], channel3[5120:15360], channel4[5120:15360], channel5[5120:15360], channel6[5120:15360], channel7[5120:15360], channel8[5120:15360]]    # 10s to 30s
                    dataStore.append(lis)
                    labels.append(0)
                    lis = [channel1[20480:], channel2[20480:], channel3[20480:], channel4[20480:], channel5[20480:], channel6[20480:], channel7[20480:], channel8[20480:]]    # 40s to 60s
                    dataStore.append(lis)
                    labels.append(1)
        return dataStore,labels

    def loadData_twoClasses_thirdarmmovement(self):
        dataStore = []
        labels = []
        for filename in os.listdir(self.dataDirectory):
            if filename.endswith('.npy'):
                file_data = self.load_data(os.path.join(self.dataDirectory, filename))
                parts = filename.split('_')
                print(parts)
                cl = parts[2]
                if cl == "2":
                    channel1 = file_data[0].astype(int)-128
                    channel2 = file_data[1].astype(int)-128
                    channel3 = file_data[2].astype(int)-128
                    channel4 = file_data[3].astype(int)-128
                    channel5 = file_data[4].astype(int)-128
                    channel6 = file_data[5].astype(int)-128
                    channel7 = file_data[6].astype(int)-128
                    channel8 = file_data[7].astype(int)-128
                    lis = [channel1[5120:15360], channel2[5120:15360], channel3[5120:15360], channel4[5120:15360], channel5[5120:15360], channel6[5120:15360], channel7[5120:15360], channel8[5120:15360]]    # 10s to 30s
                    dataStore.append(lis)
                    labels.append(0)
                    lis = [channel1[20480:], channel2[20480:], channel3[20480:], channel4[20480:], channel5[20480:], channel6[20480:], channel7[20480:], channel8[20480:]]    # 40s to 60s
                    dataStore.append(lis)
                    labels.append(1)
        return dataStore,labels