from Tackle import repointcloud, KDEAN
import numpy as np
import os
import mne
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from gudhi.point_cloud.dtm import DistanceToMeasure
import pandas as pd
import time


MIT_Fs = 256


def save_list_to_excel(data, file_name):
    df = pd.DataFrame(data).T
    # Save Excel
    df.to_excel(file_name, index=False, header=False)


def data_process(data, channel):
    chb_data = data[channel]
    DTM_mean = []
    DTM_normal_mean = []
    '''
    plt.plot(np.arange(len(chb_data)), chb_data)
    plt.show()
    print(1)
    '''
    # Slices, 4s in a section
    SLICE = []
    for j in range(int(len(chb_data) / (MIT_Fs * 4))):
        SLICE2 = []
        for jj in range(MIT_Fs * 4):
            SLICE2.append(chb_data[jj + j*(MIT_Fs * 4)])
        SLICE.append(SLICE2)

    for t in range(len(SLICE)):
        # # only for time complexity testing
        # start_time = time.time()

        pointcloud = repointcloud(len(SLICE[t]), SLICE[t], 15, 8)
        pointcloud_f = KDEAN(pointcloud, 5.2, 10)

        # normalised data
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(pointcloud)

        pointcloud_index = pointcloud_f[1]
        outpoints2 = []
        outpoints = []
        for j in range(len(normalized_data)):
            for i2 in range(len(pointcloud_index)):
                if pointcloud_index[i2] == j:
                    outpoints2.append(normalized_data[j])
                    outpoints.append(pointcloud[j])

        q = int(len(pointcloud_index)/20)+1
        dtm = DistanceToMeasure(q)
        # pointcloud = np.array(pointcloud)
        outpoints2 = np.array(outpoints2)
        outpoints = np.array(outpoints)
        DTM_values = list(dtm.fit_transform(outpoints2))
        DTM_values2 = list(dtm.fit_transform(outpoints))

        DTM_normal_mean.append(np.mean(DTM_values))
        DTM_mean.append(np.mean(DTM_values2))

        # # only for time complexity testing
        # end_time = time.time()
        # execution_time = end_time - start_time
        # print(f"Execution time: {execution_time} seconds")

    '''
    print(1)
    plt.plot(np.arange(len(DTM_mean)), DTM_mean)
    plt.show()
    '''
    return [DTM_mean, DTM_normal_mean]


def read_files(path, channel):
    files = os.listdir(path)

    edf_files = [f for f in files if f.endswith('.edf')]

    DTM_1 = []
    DTM_2 = []

    for edf_file in edf_files:
        file_path = os.path.join(path, edf_file)

        raw = mne.io.read_raw_edf(file_path, preload=True)
        raw.filter(0.5, 30)
        info = raw.info
        data, times = raw[:]
        # raw.plot(n_channels=10, duration=5, start=0)

        DTM_value = data_process(data, channel)

        DTM_1.append(DTM_value[1])
        DTM_2.append(DTM_value[0])

    # Save as Excel
    save_list_to_excel(DTM_1, 'DTM_1.xlsx')
    save_list_to_excel(DTM_2, 'DTM_2.xlsx')


# Specify folder path
chb23_path = './CHB_MIT/chb23'

read_files(chb23_path, 5)






