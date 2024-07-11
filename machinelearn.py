import pandas as pd
import scipy.stats as stats
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import os
import numpy as np
import random
import cliffs_delta


# Mann-Whitney U Test
def U_Test(X_t, Y_t):

    o_number = []
    i_number = []
    for i in range(len(Y_t)):
        if Y_t[i] == 0:
            o_number.append(i)
        if Y_t[i] == 1:
            i_number.append(i)

    sample_A = []
    sample_B = []

    for j in range(len(o_number)):
        sample_A.append(X_t[o_number[j]][0])

    for j2 in range(len(i_number)):
        sample_B.append(X_t[i_number[j2]][0])

    # Mann-Whitney U test
    u_statistic, p_value = stats.mannwhitneyu(sample_A, sample_B)

    print(f"P_value: {p_value}")
    print(f"U_value: {u_statistic}")

    # Cliff's Delta
    delta, magnitude = cliffs_delta.cliffs_delta(sample_A, sample_B)

    print(f"Cliff's Delta: {delta}")
    print(f"Effect Size: {magnitude}")


# This code is for use when testing the MIT dataset
def process_excel(file_path):

    df = pd.read_excel(file_path, header=None)
    column1 = df[0]
    column2 = df[1].dropna()

    column2_size = len(column2)

    random.seed(1000)
    sampled_column1 = random.sample(list(column1), column2_size)

    combined_list = [[val] for val in sampled_column1] + [[val] for val in column2]

    return combined_list


file_path = './DTM_values/MIT/02/DTM_2.xlsx'  # You can change the path to select the tester number
X_MIT = process_excel(file_path)

y_MIT = []
for i in range(int(len(X_MIT) / 2)):
    y_MIT.append(0)
for i in range(int(len(X_MIT) / 2)):
    y_MIT.append(1)


# This code is for use when testing the Bonn dataset
def process_excel_files(file_paths):
    combined_list = []
    for file_path in file_paths:
        df = pd.read_excel(file_path, header=None)
        num_columns = df.shape[1]
        for j in range(num_columns):
            column_data = df[j].dropna()  # NaN is removed
            list_data = [[val] for val in column_data]
            combined_list.append(list_data)
    return combined_list


def process_folders(main_folder, sub_folders):
    all_data = {}
    for folder in sub_folders:
        folder_path = os.path.join(main_folder, folder)
        file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.xlsx')]
        # if len(file_paths) != 2:
        #     raise ValueError(f"Folder {folder} does not contain exactly two Excel files.")
        all_data[folder] = process_excel_files(file_paths)
    return all_data


main_folder = './DTM_values/Bonn'
sub_folders = ['A_DTM', 'B_DTM', 'C_DTM', 'D_DTM', 'E_DTM']
result = process_folders(main_folder, sub_folders)

X_interictal_DTM1 = result['C_DTM'][0] + result['D_DTM'][0]
X_epilepsy_DTM1 = result['E_DTM'][0]
X_normal_DTM1 = result['A_DTM'][0] + result['B_DTM'][0]

X_interictal_DTM2 = result['C_DTM'][1] + result['D_DTM'][1]
X_epilepsy_DTM2 = result['E_DTM'][1]
X_normal_DTM2 = result['A_DTM'][1] + result['B_DTM'][1]

y_interictal_vs_normal = np.zeros(400)
for i in range(200):
    y_interictal_vs_normal[i] = 1

y_interictal_or_normal_vs_epilepsy = np.zeros(300)
for i in range(200):
    y_interictal_or_normal_vs_epilepsy[i] = 1


# This code is for use when testing the Iowa_State dataset
main_folder_PD = './DTM_values/Iowa_State'
sub_folders_PD = ['normal_DTM', 'PD_DTM', 'PDFOG_DTM']
result_PD = process_folders(main_folder_PD, sub_folders_PD)

normal_DTM = result_PD['normal_DTM']
PD_DTM = result_PD['PD_DTM']
PDFOG_DTM = result_PD['PDFOG_DTM']


def PD_combine(data):
    DTM1 = []
    DTM2 = []
    DTM1_average = []
    DTM2_average = []
    for i in range(int(len(data) / 4)):
        DTM1 = DTM1 + data[i]
        DTM1_average = DTM1_average + data[i + int(len(data) / 4)]
        DTM2 = DTM2 + data[i + int(len(data) / 2)]
        DTM2_average = DTM2_average + data[i + int(len(data) / 2) + int(len(data) / 4)]
    return [DTM1, DTM2, DTM1_average, DTM2_average]


X_normal_DTM1_P = PD_combine(normal_DTM)[0]
X_normal_DTM2_P = PD_combine(normal_DTM)[1]

X_PD_DTM1 = PD_combine(PD_DTM)[0]
X_PD_DTM2 = PD_combine(PD_DTM)[1]

X_PDFOG_DTM1 = PD_combine(PDFOG_DTM)[0]
X_PDFOG_DTM2 = PD_combine(PDFOG_DTM)[1]


def generate_y(X1, X2):
    y = np.zeros(int(len(X1) + len(X2)))
    for n in range(int(len(X1))):
        y[n] = 1
    return y


y_PDFOG_vs_normal = generate_y(X_PDFOG_DTM1, X_normal_DTM1_P)
y_PD_vs_normal = generate_y(X_PD_DTM1, X_normal_DTM1_P)


# This code is for use when testing the Iowa_State dataset, you can choose DTM1 or DTM2 to use
main_folder_AD_open_O2 = './DTM_values/Florida_State/AD/open/O2/DTM2.xlsx'
main_folder_HC_open_O2 = './DTM_values/Florida_State/HC/open/O2/DTM2.xlsx'
main_folder_AD_close_O2 = './DTM_values/Florida_State/AD/close/O2/DTM2.xlsx'
main_folder_HC_close_O2 = './DTM_values/Florida_State/HC/close/O2/DTM2.xlsx'

main_folder_AD_open_T6 = './DTM_values/Florida_State/AD/open/T6/DTM2.xlsx'
main_folder_HC_open_T6 = './DTM_values/Florida_State/HC/open/T6/DTM2.xlsx'
main_folder_AD_close_T6 = './DTM_values/Florida_State/AD/close/T6/DTM2.xlsx'
main_folder_HC_close_T6 = './DTM_values/Florida_State/HC/close/T6/DTM2.xlsx'

data_AD_open_O2 = pd.read_excel(main_folder_AD_open_O2, header=None).values.tolist()
data_HC_open_O2 = pd.read_excel(main_folder_HC_open_O2, header=None).values.tolist()
data_AD_close_O2 = pd.read_excel(main_folder_AD_close_O2, header=None).values.tolist()
data_HC_close_O2 = pd.read_excel(main_folder_HC_close_O2, header=None).values.tolist()

data_AD_open_T6 = pd.read_excel(main_folder_AD_open_T6, header=None).values.tolist()
data_HC_open_T6 = pd.read_excel(main_folder_HC_open_T6, header=None).values.tolist()
data_AD_close_T6 = pd.read_excel(main_folder_AD_close_T6, header=None).values.tolist()
data_HC_close_T6 = pd.read_excel(main_folder_HC_close_T6, header=None).values.tolist()


def choose(data):
    data_0 = []
    for i in range(6):
        data_0 = data_0 + data
    return data_0


y_AD = np.zeros(72+80)
for i in range(72):
    y_AD[i] = 1


def machine_learning(X, y):
    classifiers = {
        "Logistic Regression": LogisticRegression(solver='liblinear', max_iter=200),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Support Vector Machine (rbf)": SVC(kernel='rbf'),  # you can change it (rbf or linear)
        "Support Vector Machine (linear)": SVC(kernel='linear'),  # you can change it (rbf or linear)
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=20)
    }

    np.random.seed(None)
    random_seed = np.random.randint(0, 10000)
    # Define 10-fold cross validation
    kf = KFold(n_splits=10, shuffle=True, random_state=random_seed)

    results = {}

    # Please change the corresponding x and y according to the different database tests.
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X, y, cv=kf)
        results[name] = scores
        print(f"{name} Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

    # results
    results_df = pd.DataFrame(results)
    print("\nCross-Validation Results:")
    print(results_df)


machine_learning(X_MIT, y_MIT)  # Test MIT_dataset (You can change the path to select the tester number)

machine_learning(X_normal_DTM2 + X_epilepsy_DTM2, y_interictal_or_normal_vs_epilepsy)  # Test Bonn_dataset(A+B VS E)
machine_learning(X_interictal_DTM2 + X_epilepsy_DTM2, y_interictal_or_normal_vs_epilepsy)  # Test Bonn_dataset(C+D VS E)
machine_learning(X_normal_DTM1 + X_interictal_DTM1, y_interictal_vs_normal)  # Test Bonn_dataset(A+B VS C+D)

machine_learning(X_PD_DTM1 + X_normal_DTM1_P, y_PD_vs_normal)  # Test Iowa_State_dataset(PD VS normal)
machine_learning(X_PDFOG_DTM1 + X_normal_DTM1_P, y_PDFOG_vs_normal)  # Test Iowa_State_dataset(PDFOG VS normal)

machine_learning(choose(data_HC_close_T6) + data_AD_close_T6, y_AD)  # Test Floride_State_dataset(close and T6)
machine_learning(choose(data_HC_close_O2) + data_AD_close_O2, y_AD)  # Test Floride_State_dataset(close and O2)
machine_learning(choose(data_HC_open_T6) + data_AD_open_T6, y_AD)  # Test Floride_State_dataset(open and T6)
machine_learning(choose(data_HC_open_O2) + data_AD_open_O2, y_AD)  # Test Floride_State_dataset(open and O2)

U_Test(X_MIT, y_MIT)

U_Test(X_normal_DTM2 + X_epilepsy_DTM2, y_interictal_or_normal_vs_epilepsy)
U_Test(X_interictal_DTM2 + X_epilepsy_DTM2, y_interictal_or_normal_vs_epilepsy)
U_Test(X_normal_DTM1 + X_interictal_DTM1, y_interictal_vs_normal)

U_Test(X_PD_DTM1 + X_normal_DTM1_P, y_PD_vs_normal)
U_Test(X_PDFOG_DTM1 + X_normal_DTM1_P, y_PDFOG_vs_normal)

U_Test(choose(data_HC_close_T6) + data_AD_close_T6, y_AD)
U_Test(choose(data_HC_open_O2) + data_AD_open_O2, y_AD)




