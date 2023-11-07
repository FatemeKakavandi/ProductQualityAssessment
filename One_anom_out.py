import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
# Set random seed for reproducibility
np.random.seed(0)
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from src_code import OCSVM, OCSVM_train, OCSVM_test, model_results,zero_mean_normalize,df_to_array,merg_fcn,mic_f1,mac_f1,arr_2_df
import os
import pandas as pd

n_normal_samples = 100  # Number of  normal samples
n_abnormal_samples = 10 # Number of abnormal samples
n_features = 100  # Length of time series data


# Function to generate normal sine wave data
def generate_normal_sine_data(n_samples, n_features, frequency, amplitude, noise_std_dev):
    time = np.linspace(0, 10, n_features)  # Time points
    normal_data = []
    for _ in range(n_samples):
        sine_wave = amplitude * np.sin(2 * np.pi * frequency * time) + np.random.normal(0, noise_std_dev, n_features)
        normal_data.append(sine_wave)
    return np.array(normal_data)


# Function to generate abnormal sine wave data
def generate_abnormal_sine_data(n_samples, n_features, frequency, amplitude, noise_std_dev):
    time = np.linspace(0, 10, n_features)  # Time points
    abnormal_data = []
    for _ in range(n_samples):
        sine_wave = amplitude * np.sin(2 * np.pi * frequency * time) + np.random.normal(0, noise_std_dev, n_features)

        # Introduce an abnormal pattern
        abnormal_data.append(sine_wave)

    abnormal_data = np.array(abnormal_data)
    abnormal_data[:, :n_features // 10] += 2.0  # Example: Add an abnormal spike in the first 10% of data

    return abnormal_data


def stor_perf_df(pref_df,name):
    pref_df.to_excel(os.path.join('./results', name+'.xlsx'), float_format="%.4f")

## Replace your data here
base_data = generate_normal_sine_data(n_normal_samples, n_features, frequency=1.0, amplitude=1.0, noise_std_dev=0.1)
faulty_data = generate_abnormal_sine_data(n_abnormal_samples, n_features, frequency=1.0, amplitude=1.0, noise_std_dev=0.1)

y_nomral = [1]*n_normal_samples

#Plotting Data:
data = np.vstack((base_data, faulty_data))

plt.figure(figsize=(10, 6))
for sample in data:
    plt.plot(sample, color='blue' if sample in base_data else 'red', alpha=0.5)
plt.axvline(n_features // 10, color='gray', linestyle='--', label='Abnormal Pattern Start')
plt.title('Stacked Sine Wave Data (Samples on Rows, Features on Columns)')
plt.xlabel('Time')
plt.ylabel('Value')

# Normalizing the data
#base_data = zero_mean_normalize(base_data)
#faulty_data = zero_mean_normalize(faulty_data)


# Normal data train and val and test
X_train, X_test, y_train, y_test = train_test_split(base_data,y_nomral,test_size=0.15)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.15)

Nu_list = np.linspace(0.01,0.1,19,endpoint=True)
Nu_list = Nu_list.round(decimals=4)

y_val = y_val + [-1]*(n_abnormal_samples-1)
y_test = y_test+[-1]

column_list = [str(n)+'out' for n in range(0,n_abnormal_samples)]


# Combine the normal and abnormal data
#data = np.vstack((base_data, faulty_data))
#label = [0]*n_normal_samples + [1]*n_abnormal_samples

# Shuffle the data
#X_train, y_train = shuffle(X_train, y_train)
#X_test, y_test = shuffle(X_test, y_test)
#X_val, y_val = shuffle(X_val, y_val)


F1_val = []
recall_val = []
precision_val = []

F1_test = []
recall_test = []
precision_test = []

mic_f1_arr =[]
mac_f1_arr = []
mic_f1_arr_test = []
mac_f1_arr_test = []

for nu in Nu_list:
    model = OCSVM(nu)
    model.fit(X_train)
    f1_nu = []
    recal_nu = []
    pre_nu = []

    f1_test_nu = []
    recal_test_nu = []
    pre_test_nu = []

    mic_f1_nu =[]
    mac_f1_nu=[]

    mic_f1_nu_test = []
    mac_f1_nu_test = []

    i=1
    for sample in range(n_abnormal_samples):
        # Validation part
        temp_fault = np.delete(faulty_data,sample,0)
        val_set = np.vstack((X_val,temp_fault))
        y_hat = model.predict(val_set)

        f1_val, recal_val, pre_val = model_results(y_val, y_hat)
        # Validation Micro and Macro F1
        mic_f1_nu = mic_f1_nu + [mic_f1(y_val, y_hat)]
        mac_f1_nu = mac_f1_nu + [mac_f1(y_val, y_hat)]

        f1_nu = f1_nu + [f1_val]
        recal_nu = recal_nu + [recal_val]
        pre_nu = pre_nu + [pre_val]

        # Testing time
        test_arr = np.vstack((X_test,faulty_data[sample,:]))
        y_test_hat = model.predict(test_arr)
        f1_test, recal_test, pre_test = model_results(y_test,y_test_hat)

        # Test Micro and Macro F1
        mic_f1_nu_test = mic_f1_nu_test + [mic_f1(y_test,y_test_hat)]
        mac_f1_nu_test = mac_f1_nu_test + [mac_f1(y_test,y_test_hat)]


        f1_test_nu = f1_test_nu + [f1_test]
        recal_test_nu = recal_test_nu + [recal_test]
        pre_test_nu = pre_test_nu + [pre_test]

        #F1_test._set_value(nu,str(i)+'out',f1_test)
        #recall_test._set_value(nu, str(i) + 'out', recal_test)
        #precision_test._set_value(nu, str(i) + 'out', pre_test)
        i = i +1
    F1_val.append(f1_nu)
    recall_val.append(recal_nu)
    precision_val.append(pre_nu)

    F1_test.append(f1_test_nu)
    recall_test.append(recal_test_nu)
    precision_test.append((pre_test_nu))

    mic_f1_arr.append(mic_f1_nu)
    mac_f1_arr.append(mac_f1_nu)
    mic_f1_arr_test.append(mic_f1_nu_test)
    mac_f1_arr_test.append((mac_f1_nu_test))


#f1_val_df = arr_2_df(F1_val,column_list, Nu_list)
#recal_val_df =arr_2_df(recall_val,column_list, Nu_list)
#pre_val_df = arr_2_df(precision_val,column_list,Nu_list)

#f1_test_df = arr_2_df(F1_test, column_list, Nu_list)
#recal_test_df = arr_2_df(recall_test, column_list, Nu_list)
#pre_test_df = arr_2_df(precision_test, column_list, Nu_list)

mic_f1_df = arr_2_df(mic_f1_arr,column_list,Nu_list)
mac_f1_df = arr_2_df(mac_f1_arr, column_list, Nu_list)

mic_f1_test_df = arr_2_df(mic_f1_arr_test,column_list,Nu_list)
mac_f1_test_df = arr_2_df(mac_f1_arr_test,column_list,Nu_list)


stor_perf_df(mic_f1_df,'Mic_F1_Val')
stor_perf_df(mac_f1_df,'Mac_F1_Val')
stor_perf_df(mic_f1_test_df,'Mic_F1_Test')
stor_perf_df(mac_f1_test_df,'Mac_F1_Test')
#return f1_val_df, recal_val_df,pre_val_df ,f1_test_df ,recal_test_df, pre_test_df,mic_f1_df,mac_f1_df#F1_val,recall_val,precision_val,F1_test,recall_test, precision_test

plt.figure()
plt.bar(np.arange(len(mic_f1_df)),np.mean(mic_f1_df,axis=1))
plt.bar(np.arange(len(mic_f1_test_df)),np.mean(mic_f1_test_df,axis=1))

plt.show()