import numpy as np
import random
from gudhi.point_cloud.dtm import DistanceToMeasure
import os
from sklearn.preprocessing import MinMaxScaler
from Tackle import filter_wave, repointcloud, KDEAN
import pandas as pd
import time

'''
First, it's Bonn_dataset
'''


def txt_to_data(a):
    file_contents = []
    for filename in os.listdir(a):
        file_path = os.path.join(a, filename)

        if os.path.isfile(file_path):
            with open(file_path, "r") as file:

                temp_content = []
                for line in file:
                    temp_content.append(line.strip())

                file_contents.append(temp_content)

    float_contents = []

    for file_index, content in enumerate(file_contents):
        float_content = []

        for line in content:
            try:
                float_value = int(line)
                float_content.append(float_value)
            except ValueError:
                print(f"Unable to convert line {line} to float in file {file_index}")

        float_contents.append(float_content)
    return float_contents


# List all files in the folder
folder_path0 = 'Bonn_seizures/A/Z'
folder_path1 = 'Bonn_seizures/B/O'
folder_path2 = 'Bonn_seizures/C/N'
folder_path3 = 'Bonn_seizures/D/F'
folder_path4 = 'Bonn_seizures/E/S'

A = txt_to_data(folder_path0)  # Normal (eyes close)
B = txt_to_data(folder_path1)  # Normal (eyes open)
C = txt_to_data(folder_path2)  # interictal period
D = txt_to_data(folder_path3)  # interictal period（seizures size）
E = txt_to_data(folder_path4)  # seizure period

Use_drawing = []


def calculate(group, Time, band):

    AL = list(filter_wave(group[Time], 173.61))
    Slice = list(range(5))

    for i in range(5):
        Slice[i] = (AL[i*800:((i+1)*800)])

    save_data = []
    save_data2 = []
    index_number = []

    for t in range(4):
        # # only for time complexity testing
        # start_time = time.time()

        point_cloud = repointcloud(len(Slice[t]), Slice[t], 5, 8)
        Use_drawing.append(point_cloud)

        point_cloud_filter = KDEAN(point_cloud, band, 5)

        # normalised data
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(point_cloud)

        Record_index = point_cloud_filter[1]
        index_number.append(len(Record_index))
        outpoints2 = []
        outpoints = []
        for j in range(len(normalized_data)):
            for i2 in range(len(Record_index)):
                if Record_index[i2] == j:
                    outpoints2.append(normalized_data[j])
                    outpoints.append(point_cloud[j])

        q = int(len(Record_index)/20)+1
        dtm = DistanceToMeasure(q)

        outpoints2 = np.array(outpoints2)
        outpoints = np.array(outpoints)
        DTM_values = list(dtm.fit_transform(outpoints2))
        DTM_values2 = list(dtm.fit_transform(outpoints))

        # # only for time complexity testing
        # end_time = time.time()
        # execution_time = end_time - start_time
        # print(f"Execution time: {execution_time} seconds")

        DTM_values.sort(reverse=True)
        DTM_values2.sort(reverse=True)

        DTM_average = sum(DTM_values[0:len(Record_index)]) / len(Record_index)
        save_data.append(DTM_average)
        DTM2_average = sum(DTM_values2[0:len(Record_index)]) / len(Record_index)
        save_data2.append(DTM2_average)

    return [save_data, save_data2]


# Test
# Result = calculate(A, 27, 3.00)

# Calculate the average DTM values D1 and D2
def DTM_value_calculate(group, band):
    DTM_1 = []
    DTM_2 = []

    for t in range(100):
        S = calculate(group, t, band)
        DTM1 = sum(S[0]) / len(S[0])
        DTM2 = sum(S[1]) / len(S[1])

        DTM_1.append(DTM1)
        DTM_2.append(DTM2)

    return [DTM_1, DTM_2]

# Test
# Ade = DTM_value_calculate(A, 3.00)[1]


A_DTM = DTM_value_calculate(A, 3.00)
B_DTM = DTM_value_calculate(B, 3.00)
C_DTM = DTM_value_calculate(C, 3.00)
D_DTM = DTM_value_calculate(D, 3.00)
E_DTM = DTM_value_calculate(E, 3.00)


lists = {
    'A_DTM': A_DTM,
    'B_DTM': B_DTM,
    'C_DTM': C_DTM,
    'D_DTM': D_DTM,
    'E_DTM': E_DTM
}

# Specify the home folder
main_folder = './DTM_values/Bonn'

for list_name, dtm_list in lists.items():

    sub_folder = os.path.join(main_folder, list_name)
    os.makedirs(sub_folder, exist_ok=True)

    for i, sub_list in enumerate(dtm_list):
        df = pd.DataFrame(sub_list)
        excel_file = os.path.join(sub_folder, f'DTM_{i + 1}.xlsx')
        df.to_excel(excel_file, index=False)


'''
Second, it's Iowa_State_dataset
'''

file_path_Parkinson = 'Parkinson/PD'
file_path_normal = 'Parkinson/normal'
file_path_ParkinsonFOG = 'Parkinson/PDFOG'

files_P = os.listdir(file_path_Parkinson)
files_n = os.listdir(file_path_normal)
files_PF = os.listdir(file_path_ParkinsonFOG)


def read_data(FILE, PATH):
    all_data = []
    for files in FILE:

        file_path_PD = os.path.join(PATH, files)

        with open(file_path_PD, 'r') as file:
            # Reads the contents of a file and stores it in a list
            content_list = file.readlines()

            data = []
            for line in content_list:
                data.append(line.strip())
            a_float = []
            for num in data:
                a_float.append(float(num))
            all_data.append(a_float)

    return all_data


data_PD = read_data(files_P, file_path_Parkinson)
data_normal = read_data(files_n, file_path_normal)
data_PDFOG = read_data(files_PF, file_path_ParkinsonFOG)


def calculate_Parkinson_or_AD(data, slice_nu, T, M, band):

    all_Parkinson_or_AD_DTM = []
    all_Parkinson_or_AD_DTM2 = []

    for y in range(len(data)):
        SLICE = []
        for j in range(int(len(data[y])/slice_nu)):
            SLICE.append(data[y][j*slice_nu:(j+1)*slice_nu])

        DTM_Parkinson_or_AD = []
        DTM_Parkinson_or_AD_2 = []

        for i in range(len(SLICE)):
            # # only for time complexity testing
            # start_time = time.time()

            point_cloud = repointcloud(len(SLICE[i]), SLICE[i], T, M)
            CC = KDEAN(point_cloud, band, 10)

            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(point_cloud)

            ZZ = CC[1]
            outpoints2 = []
            outpoints = []
            for j in range(len(normalized_data)):
                for i2 in range(len(ZZ)):
                    if ZZ[i2] == j:
                        outpoints2.append(normalized_data[j])
                        outpoints.append(point_cloud[j])

            q = int(len(ZZ) / 20) + 1
            dtm = DistanceToMeasure(q)

            outpoints2 = np.array(outpoints2)
            outpoints = np.array(outpoints)
            DTM_values = list(dtm.fit_transform(outpoints2))
            DTM_values2 = list(dtm.fit_transform(outpoints))

            DTM_values.sort(reverse=True)
            DTM_values2.sort(reverse=True)
            DTM1_mean = sum(DTM_values[0:len(ZZ)]) / len(ZZ)
            DTM_Parkinson_or_AD.append(DTM1_mean)
            DTM2_mean = sum(DTM_values2[0:len(ZZ)]) / len(ZZ)
            DTM_Parkinson_or_AD_2.append(DTM2_mean)

            # # only for time complexity testing
            # end_time = time.time()
            # execution_time = end_time - start_time
            # print(f"Execution time: {execution_time} seconds")

        all_Parkinson_or_AD_DTM.append(DTM_Parkinson_or_AD)
        all_Parkinson_or_AD_DTM2.append(DTM_Parkinson_or_AD_2)

    return [all_Parkinson_or_AD_DTM, all_Parkinson_or_AD_DTM2]


normal_DTM = calculate_Parkinson_or_AD(data_normal, 2000, 4, 12, 3.6)
PD_DTM = calculate_Parkinson_or_AD(data_PD, 2000, 4, 12, 3.6)
PDFOG_DTM = calculate_Parkinson_or_AD(data_PDFOG, 2000, 4, 12, 3.6)


# data save
def save_dtm_to_excel(main_folder, dtm_name, dtm_data):

    folder_path = os.path.join(main_folder, dtm_name)
    os.makedirs(folder_path, exist_ok=True)

    for i, sublist in enumerate(dtm_data):
        df = pd.DataFrame(sublist).T
        excel_file = os.path.join(folder_path, f'DTM{i + 1}.xlsx')
        df.to_excel(excel_file, index=False)


main_folder = './DTM_values/Iowa_State'

save_dtm_to_excel(main_folder, 'normal_DTM', normal_DTM)
save_dtm_to_excel(main_folder, 'PD_DTM', PD_DTM)
save_dtm_to_excel(main_folder, 'PDFOG_DTM', PDFOG_DTM)


'''
Third, it's Florida_State_dataset
'''

file_path_AD_close = './osfstorage-archive/EEG_data/AD/Eyes_closed'
file_path_AD_open = './osfstorage-archive/EEG_data/AD/Eyes_open'
file_path_HC_close = './osfstorage-archive/EEG_data/Healthy/Eyes_closed'
file_path_HC_open = './osfstorage-archive/EEG_data/Healthy/Eyes_open'


def process(fp, shu):
    patient = []
    for i in range(1, shu):
        # Generate folder name based on patient number
        paciente_folder_name = f'Paciente{i}'  # such as Paciente1、Paciente2、...、Paciente80
        paciente_folder_path = os.path.join(fp, paciente_folder_name)
        if os.path.exists(paciente_folder_path):
            Save = []
            for filename in os.listdir(paciente_folder_path):
                if filename == 'O2.txt' or filename == 'T6.txt':
                    txt_filepath = os.path.join(paciente_folder_path, filename)
                    with open(txt_filepath, 'r') as file:
                        data_list = [float(value) for value in file.read().split()]
                        Save.append(calculate_Parkinson_or_AD([data_list], 1024, 19, 6, 4.2))
            patient.append(Save)
    return patient


all_data_AD_close = process(file_path_AD_close, 81)
all_data_HC_close = process(file_path_HC_close, 13)
all_data_AD_open = process(file_path_AD_open, 81)
all_data_HC_open = process(file_path_HC_open, 13)


def insert(X):
    DTM1_O2 = []
    DTM2_O2 = []
    DTM1_T6 = []
    DTM2_T6= []

    for i in range(len(X)):
        DTM1_O2.append(X[i][0][0][0][0])
        DTM2_O2.append(X[i][0][1][0][0])

        DTM1_T6.append(X[i][1][0][0][0])
        DTM2_T6.append(X[i][1][1][0][0])

    return [DTM1_O2, DTM2_O2, DTM1_T6, DTM2_T6]


AD_close_DTM1_O2 = insert(all_data_AD_close)[0]
AD_close_DTM1_T6 = insert(all_data_AD_close)[2]
AD_close_DTM2_O2 = insert(all_data_AD_close)[1]
AD_close_DTM2_T6 = insert(all_data_AD_close)[3]

AD_open_DTM1_O2 = insert(all_data_AD_open)[0]
AD_open_DTM1_T6 = insert(all_data_AD_open)[2]
AD_open_DTM2_O2 = insert(all_data_AD_open)[1]
AD_open_DTM2_T6 = insert(all_data_AD_open)[3]

HC_close_DTM1_O2 = insert(all_data_HC_close)[0]
HC_close_DTM1_T6 = insert(all_data_HC_close)[2]
HC_close_DTM2_O2 = insert(all_data_HC_close)[1]
HC_close_DTM2_T6 = insert(all_data_HC_close)[3]

HC_open_DTM1_O2 = insert(all_data_HC_open)[0]
HC_open_DTM1_T6 = insert(all_data_HC_open)[2]
HC_open_DTM2_O2 = insert(all_data_HC_open)[1]
HC_open_DTM2_T6 = insert(all_data_HC_open)[3]

# save data
main_folder_AD_open_O2 = './DTM_values/Florida_State/AD/open/O2'
main_folder_HC_open_O2 = './DTM_values/Florida_State/HC/open/O2'
main_folder_AD_close_O2 = './DTM_values/Florida_State/AD/close/O2'
main_folder_HC_close_O2 = './DTM_values/Florida_State/HC/close/O2'

main_folder_AD_open_T6 = './DTM_values/Florida_State/AD/open/T6'
main_folder_HC_open_T6 = './DTM_values/Florida_State/HC/open/T6'
main_folder_AD_close_T6 = './DTM_values/Florida_State/AD/close/T6'
main_folder_HC_close_T6 = './DTM_values/Florida_State/HC/close/T6'


def save_simple(data, path, i):
    df = pd.DataFrame(data)
    excel_file = os.path.join(path, f'DTM{i + 1}.xlsx')
    df.to_excel(excel_file, index=False)


save_simple(AD_close_DTM1_O2, main_folder_AD_close_O2, 0)
save_simple(AD_close_DTM1_T6, main_folder_AD_close_T6, 0)
save_simple(AD_close_DTM2_O2, main_folder_AD_close_O2, 1)
save_simple(AD_close_DTM2_T6, main_folder_AD_close_T6, 1)

save_simple(AD_open_DTM1_O2, main_folder_AD_open_O2, 0)
save_simple(AD_open_DTM1_T6, main_folder_AD_open_T6, 0)
save_simple(AD_open_DTM2_O2, main_folder_AD_open_O2, 1)
save_simple(AD_open_DTM2_T6, main_folder_AD_open_T6, 1)

save_simple(HC_close_DTM1_O2, main_folder_HC_close_O2, 0)
save_simple(HC_close_DTM1_T6, main_folder_HC_close_T6, 0)
save_simple(HC_close_DTM2_O2, main_folder_HC_close_O2, 1)
save_simple(HC_close_DTM2_T6, main_folder_HC_close_T6, 1)

save_simple(HC_open_DTM1_O2, main_folder_HC_open_O2, 0)
save_simple(HC_open_DTM1_T6, main_folder_HC_open_T6, 0)
save_simple(HC_open_DTM2_O2, main_folder_HC_open_O2, 1)
save_simple(HC_open_DTM2_T6, main_folder_HC_open_T6, 1)

